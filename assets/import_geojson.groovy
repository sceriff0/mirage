/**
 * ============================================================================
 * QUPATH SCRIPT: Import GeoJSON Cell Detections (fast, with load progress)
 * ============================================================================
 *
 * Imports cells.geojson (from Mirage's EXPORT_GEOJSON) into the open image
 * using QuPath's NATIVE reader (PathIO.readObjects). This is dramatically
 * faster than drag-and-drop or hand-parsing the JSON, and — unlike a manual
 * parser — it faithfully reconstructs `cell` objects together with their
 * `nucleusGeometry` and raw measurement names (CD45, MORPH: Area µm², ...),
 * exactly as FlowPath expects.
 *
 * WHY THIS IS FASTER THAN DRAG-AND-DROP:
 *   Drag-and-drop re-parses the whole GeoJSON and resolves the hierarchy on
 *   the UI thread every single time. This script parses once with the native
 *   streaming reader, adds objects in batches (with a live % readout), and
 *   skips the costly hierarchy resolve for flat detections.
 *
 * USAGE:
 *   1. Open your pyramidal OME-TIFF (the intensity series, Image:0) in QuPath.
 *   2. Automate → Show script editor → paste this script.
 *   3. Run (Ctrl+R). Either set GEOJSON_PATH below, or pick the file when asked.
 *   4. Watch the log pane for "loaded X / Y (Z%)" progress.
 *   5. Shift+1 toggles detection visibility.
 *
 * TIP: after the first import, save the image into a QuPath PROJECT. Reopening
 *      the project loads objects from QuPath's binary .qpdata store — near
 *      instant — so you never pay the GeoJSON parse again.
 * ============================================================================
 */

import qupath.lib.io.PathIO

// ============================================================================
// CONFIGURATION
// ============================================================================

// Hardcode a path to skip the file chooser (handy for batching); else leave null.
def GEOJSON_PATH = null            // e.g. "/data/results/P001/geojson/cells.geojson"

// How many objects to add per batch. Larger = fewer repaints (faster) but
// coarser progress. 25k is a good balance for million-cell slides.
def BATCH_SIZE = 25000

// Re-parent detections under enclosing annotations. For flat cell detections
// (no annotations) this is pure cost with no benefit — leave false. Set true
// only if you have annotation regions and need cells nested inside them.
def RESOLVE_HIERARCHY = false

// ============================================================================
// HELPERS
// ============================================================================

def fmt = { long n -> String.format("%,d", n) }
def secs = { long ms -> String.format("%.1fs", ms / 1000.0) }

def registerToggle = {
    def viewer = getCurrentViewer()
    if (viewer == null) return
    def view = viewer.getView()
    def existing = view.getProperties().get("_detectionToggleHandler")
    if (existing != null) {
        view.removeEventHandler(javafx.scene.input.KeyEvent.KEY_PRESSED, existing)
        view.getProperties().remove("_detectionToggleHandler")
    }
    def visible = [val: true]
    def handler = { javafx.scene.input.KeyEvent event ->
        if (event.isShiftDown() && event.getCode() == javafx.scene.input.KeyCode.DIGIT1) {
            event.consume()
            visible.val = !visible.val
            viewer.getOverlayOptions().setShowDetections(visible.val)
            println(visible.val ? "Detections: visible" : "Detections: hidden")
        }
    } as javafx.event.EventHandler
    view.addEventHandler(javafx.scene.input.KeyEvent.KEY_PRESSED, handler)
    view.getProperties().put("_detectionToggleHandler", handler)
    println "Shift+1 -> toggle detection visibility"
}

// ============================================================================
// SCRIPT
// ============================================================================

println "=" * 64
println "QuPath GeoJSON Import (native reader, FlowPath compatible)"
println "=" * 64

// --- Guard: an image must be open --------------------------------------------
def imageData = getCurrentImageData()
if (imageData == null) {
    Dialogs.showErrorNotification("Import GeoJSON", "Open an image first (the Image:0 intensity series).")
    return
}
def server = imageData.getServer()
println "Image      : ${server.getMetadata().getName()}"

// --- Resolve the GeoJSON file (config path or file chooser) ------------------
def geojsonFile = null
if (GEOJSON_PATH != null && !GEOJSON_PATH.trim().isEmpty()) {
    geojsonFile = new File(GEOJSON_PATH)
    if (!geojsonFile.exists()) {
        Dialogs.showErrorNotification("Import GeoJSON", "File not found:\n${GEOJSON_PATH}")
        return
    }
} else {
    def latch = new java.util.concurrent.CountDownLatch(1)
    javafx.application.Platform.runLater {
        def fc = new javafx.stage.FileChooser()
        fc.setTitle("Select cells.geojson")
        fc.getExtensionFilters().add(
            new javafx.stage.FileChooser.ExtensionFilter("GeoJSON", "*.geojson", "*.json")
        )
        geojsonFile = fc.showOpenDialog(null)
        latch.countDown()
    }
    latch.await()
}
if (geojsonFile == null) {
    println "Cancelled."
    return
}
def sizeMb = geojsonFile.length() / (1024.0 * 1024.0)
println String.format("GeoJSON    : %s (%.1f MB)", geojsonFile.name, sizeMb)

// --- Offer to clear existing detections --------------------------------------
def existing = getDetectionObjects()
if (existing.size() > 0) {
    def ok = Dialogs.showConfirmDialog(
        "Existing detections",
        "Found ${fmt(existing.size())} existing detections.\nClear them and import fresh?")
    if (!ok) { println "Cancelled (kept existing detections)."; registerToggle(); return }
    println "Clearing ${fmt(existing.size())} existing detections..."
    removeObjects(existing, false)
    fireHierarchyUpdate()
}

// --- Phase 1: parse (native, single fast pass; no sub-progress available) -----
println "\nParsing GeoJSON (native reader — the heavy step for large files)..."
long t0 = System.currentTimeMillis()
def objects
try {
    objects = PathIO.readObjects(geojsonFile.toPath())
} catch (Exception e) {
    Dialogs.showErrorNotification("Import GeoJSON", "Failed to read GeoJSON: ${e.message}")
    println "ERROR: ${e}"
    return
}
long tParse = System.currentTimeMillis() - t0
int total = objects.size()
println "  parsed ${fmt(total)} objects in ${secs(tParse)}"

if (total == 0) {
    println "Nothing to import (0 features). Is this the right cells.geojson?"
    registerToggle()
    return
}

// --- Classification breakdown -------------------------------------------------
def counts = [:].withDefault { 0 }
objects.each { counts[it.getPathClass()?.toString() ?: "Unclassified"]++ }
println "Classifications:"
counts.sort { -it.value }.each { name, c ->
    println String.format("  %-20s %s (%.1f%%)", name, fmt(c), 100.0 * c / total)
}

// --- Phase 2: add in batches with live progress ------------------------------
println "\nLoading cells into the hierarchy..."
def hierarchy = getCurrentHierarchy()
long t1 = System.currentTimeMillis()
for (int i = 0; i < total; i += BATCH_SIZE) {
    int end = Math.min(i + BATCH_SIZE, total)
    hierarchy.addObjects(objects.subList(i, end))   // bulk add; repaint per batch
    println String.format("  loaded %s / %s  (%.1f%%)", fmt(end), fmt(total), 100.0 * end / total)
}
long tAdd = System.currentTimeMillis() - t1

if (RESOLVE_HIERARCHY) {
    println "Resolving hierarchy (nesting under annotations)..."
    resolveHierarchy()
}
fireHierarchyUpdate()

// --- Summary -----------------------------------------------------------------
println "\n" + "=" * 64
println "DONE: imported ${fmt(total)} cells"
println "  parse ${secs(tParse)}  |  load ${secs(tAdd)}  |  total ${secs(System.currentTimeMillis() - t0)}"
println "  Tip: save this image into a QuPath project so re-opening skips the parse."
registerToggle()
println "Open FlowPath for interactive gating."
println "=" * 64
