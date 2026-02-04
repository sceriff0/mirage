/**
 * ============================================================================
 * QUPATH SCRIPT: Import Pixie Cell Clusters
 * ============================================================================
 *
 * Imports cells from GeoJSON with Pixie cluster classifications and colors.
 * Works with output from PIXIE_CELL_CLUSTER process in the ATEIA pipeline.
 *
 * OUTPUT FILES EXPECTED (from pixie_cell_cluster.py):
 *   - pixie_clusters.geojson              <- Cell detections with cluster assignments
 *   - pixie_clusters.classifications.json <- Cluster color definitions (auto-loaded)
 *
 * USAGE:
 *   1. Open your pyramidal OME-TIFF in QuPath
 *   2. Automate -> Show script editor -> Paste this script
 *   3. Run (Ctrl+R) -> Select your pixie_clusters.geojson file
 *   4. Colors load automatically from pixie_clusters.classifications.json
 *
 * MEASUREMENTS IMPORTED:
 *   - Cluster_ID: Meta-cluster numeric ID
 *   - SOM_Cluster: SOM cluster assignment
 *   - Cell_Area: Cell area in pixels
 *   - Centroid_X/Y: Cell centroid coordinates
 *
 * ============================================================================
 */

import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClass
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.gui.dialogs.Dialogs
import com.google.gson.JsonParser

// ============================================================================
// CONFIGURATION
// ============================================================================

def CELL_RADIUS_UM = 3.0  // Cell marker radius in micrometers

// ============================================================================
// SCRIPT
// ============================================================================

println "=" * 60
println "QuPath Pixie Cluster Import"
println "=" * 60

// Check image
def imageData = getCurrentImageData()
if (imageData == null) {
    Dialogs.showErrorMessage("Error", "Please open an image first!")
    return
}

def server = imageData.getServer()
def pixelSize = server.getPixelCalibration().getAveragedPixelSizeMicrons()
println "Image: ${server.getMetadata().getName()}"
println "Pixel size: ${pixelSize} um"

// Prompt for GeoJSON file
def geojsonFile = Dialogs.promptForFile("Select Pixie Clusters GeoJSON", null, "GeoJSON", ".geojson")
if (geojsonFile == null) {
    println "Cancelled."
    return
}

println "GeoJSON: ${geojsonFile.name}"

// Look for classifications file (same name pattern as phenotypes)
def classificationsFile = new File(geojsonFile.absolutePath.replace(".geojson", ".classifications.json"))
def clusterColors = [:]

if (classificationsFile.exists()) {
    println "Found classifications: ${classificationsFile.name}"
    def classText = classificationsFile.text
    def classArray = JsonParser.parseString(classText).getAsJsonArray()
    classArray.each { item ->
        def obj = item.getAsJsonObject()
        def name = obj.get("name").getAsString()
        if (obj.has("rgb")) {
            def rgbArray = obj.get("rgb").getAsJsonArray()
            clusterColors[name] = [
                rgbArray.get(0).getAsInt(),
                rgbArray.get(1).getAsInt(),
                rgbArray.get(2).getAsInt()
            ]
        }
    }
    println "Loaded ${clusterColors.size()} cluster colors"
} else {
    println "No classifications file found, will use auto-generated colors"
}

// Load GeoJSON
println "\nLoading GeoJSON..."
def geojsonText = geojsonFile.text
def geojson = JsonParser.parseString(geojsonText).getAsJsonObject()
def features = geojson.get("features").getAsJsonArray()
println "Found ${features.size()} cells"

// Helper to get PathClass with color
def getPathClass = { String name ->
    def rgb = clusterColors[name]
    if (rgb) {
        return PathClass.fromString(name, getColorRGB(rgb[0], rgb[1], rgb[2]))
    }
    // Auto-generate color based on cluster number
    def clusterNum = name.replaceAll(/[^0-9]/, '') ?: "0"
    def hash = Math.abs(clusterNum.toInteger() * 2654435761)
    def r = ((hash >> 16) & 0xFF)
    def g = ((hash >> 8) & 0xFF)
    def b = (hash & 0xFF)
    // Ensure reasonable brightness
    if (r + g + b < 200) {
        r = Math.min(255, r + 100)
        g = Math.min(255, g + 100)
        b = Math.min(255, b + 100)
    }
    return PathClass.fromString(name, getColorRGB(r, g, b))
}

// Count clusters
def counts = [:].withDefault { 0 }
features.each { f ->
    def props = f.getAsJsonObject().get("properties")?.getAsJsonObject()
    def classification = props?.get("classification")?.getAsJsonObject()
    def name = classification?.get("name")?.getAsString() ?: "Unclassified"
    counts[name]++
}

println "\nClusters found:"
// Sort by cluster number for cleaner display
def sortedCounts = counts.sort { a, b ->
    def numA = a.key.replaceAll(/[^0-9]/, '') ?: "999"
    def numB = b.key.replaceAll(/[^0-9]/, '') ?: "999"
    numA.toInteger() <=> numB.toInteger()
}
sortedCounts.each { name, count ->
    def pct = (count / features.size() * 100).round(1)
    def rgb = clusterColors[name] ?: [128, 128, 128]
    println "  ${name}: ${count} cells (${pct}%) RGB(${rgb.join(',')})"
}

// Import cells
println "\nImporting cells..."
def plane = ImagePlane.getDefaultPlane()
def radius = CELL_RADIUS_UM / pixelSize
def detections = []

features.eachWithIndex { featureElement, idx ->
    try {
        def feature = featureElement.getAsJsonObject()
        def geometry = feature.get("geometry").getAsJsonObject()
        def coords = geometry.get("coordinates").getAsJsonArray()

        // Coordinates in GeoJSON are already in pixels - use directly
        def x_px = coords.get(0).getAsDouble()
        def y_px = coords.get(1).getAsDouble()

        def props = feature.get("properties")?.getAsJsonObject()
        def classification = props?.get("classification")?.getAsJsonObject()
        def className = classification?.get("name")?.getAsString() ?: "Unclassified"
        def pathClass = getPathClass(className)

        def roi = ROIs.createEllipseROI(
            x_px - radius, y_px - radius,
            radius * 2, radius * 2, plane
        )

        def detection = PathObjects.createDetectionObject(roi, pathClass)

        // Add measurements from properties
        def measurements = props?.get("measurements")?.getAsJsonObject()
        if (measurements != null) {
            measurements.entrySet().each { entry ->
                try {
                    detection.getMeasurementList().put(entry.key, entry.value.getAsDouble())
                } catch (Exception e) {
                    // Skip non-numeric measurements
                }
            }
        }
        detection.getMeasurementList().close()

        detections.add(detection)

        if ((idx + 1) % 25000 == 0) {
            println "  Processed ${idx + 1}/${features.size()}"
        }
    } catch (Exception e) {
        // Skip errors silently
    }
}

// Add to image
addObjects(detections)
fireHierarchyUpdate()

// Summary
println "\n" + "=" * 60
println "DONE! Imported ${detections.size()} cells with Pixie clusters"
println "=" * 60
println "\nTips:"
println "  - Press 'D' to toggle cell visibility"
println "  - Press 'F' to fill cells with cluster colors"
println "  - View -> Cell display -> Centroids only (for faster rendering)"
println "  - Measure -> Show detection measurements (to see Cluster_ID, SOM_Cluster)"
println ""
println "To rename clusters:"
println "  1. Edit cell_meta_cluster_mapping.csv"
println "  2. Re-run pixie_cell_cluster.py with --mapping option"
println "  3. Or manually reclassify in QuPath: Select cells -> Set class"
