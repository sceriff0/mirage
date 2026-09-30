#!/usr/bin/env Rscript
# =============================================================================
# benchmarks/ihc/supplementary.R  —  the supplementary figures whose data live in
# ../ihc_method (FlowPath calls, clinical categories, RNA deconvolution):
#
#   S3b  fraction of CD8+ cells that are also CD3+, per case
#   S9   every case's field coloured by FlowPath phenotype call (lineage + compartment)
#   S10  CD45+ fraction per case, individually and grouped cold/intermediate/hot
#   S11  imaging fractions vs EVERY deconvolution method, raw paired points, no fit
#
# RUN AGAINST AN ihc_method CHECKOUT, from mirage (benchmarks/submit_supplementary.sh
# does this, beside benchmarks/supplementary.py for S2/S3a/S4-S8 and the mosaic):
#
#     IHC_ROOT=../ihc_method Rscript benchmarks/ihc/supplementary.R [S3 S9 S10 S11]
#
# It lives in mirage because this is where the supplementary set is assembled; it only
# SOURCES ihc_method's builders (code/paper_figures.R, code/cell_tables.R, ...) exactly as
# figures/fig5.R does, and adds no science of its own. Copy it to ihc_method/figures/ to
# own it there instead -- it runs unchanged from either place.
#
# Writes <IHC_ROOT>/output/figures/supplementary/<S>/ : <S>_*.pdf/.png + <S>_values.csv
# (the AUTHORS TO SUPPLY numbers). A figure whose inputs are absent is skipped, loudly.
# =============================================================================

root <- Sys.getenv("IHC_ROOT", "")
if (!nzchar(root)) root <- tryCatch(here::here(), error = function(e) normalizePath("."))
root <- normalizePath(root)
if (!file.exists(file.path(root, "figures", "_common.R")))
  stop("supplementary.R: ", root, " is not an ihc_method checkout (no figures/_common.R). ",
       "Set IHC_ROOT.")
setwd(root)                                   # here::here() inside the builders
source(file.path(root, "figures", "_common.R"))
suppressPackageStartupMessages({
  library(dplyr); library(tibble); library(ggplot2)
})
source(file.path(root, "code", "validation_helpers.R"))
source(file.path(root, "code", "paper_figures.R"))
source(file.path(root, "code", "arm_cells.R"))
source(file.path(root, "code", "cell_tables.R"))

ARM            <- Sys.getenv("IHC_ARM", "massimo2")   # the phenotyping arm, as fig5.R
HOTCOLD_SOURCE <- "immuno_phe"   # S10/S11: the clinical category (cold/intermediate/hot)
OUT            <- file.path(root, "output", "figures", "supplementary")
WANT           <- commandArgs(trailingOnly = TRUE)
want <- function(s) !length(WANT) || s %in% WANT

save_supp <- function(p, fig, name, width_mm, height_mm) {
  d <- file.path(OUT, fig); dir.create(d, recursive = TRUE, showWarnings = FALSE)
  for (dev in c("pdf", "png"))
    ggsave(file.path(d, paste0(name, ".", dev)), p, width = width_mm, height = height_mm,
           units = "mm", dpi = 300,
           device = if (dev == "pdf") grDevices::cairo_pdf else "png")
  message("  ", fig, ": ", file.path(d, name))
}
write_values <- function(df, fig, name = paste0(fig, "_values")) {
  d <- file.path(OUT, fig); dir.create(d, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(df, file.path(d, paste0(name, ".csv")), row.names = FALSE)
}
skip <- function(fig, why) message("  ", fig, ": SKIPPED (", why, ")")

# --- the arm: cells, polygons, union metrics (the same sequence as figures/fig5.R) ----
as_spec  <- arm_spec(ARM)
have_arm <- dir.exists(as_spec$region_csv$path)

# --- --check: what each figure would be drawn from, WITHOUT loading a cell ----------
# Writes <OUT>/check_ihc.csv, which benchmarks/supplementary.py --ihc merges into its
# own check.csv. Every test here is a file test, so it runs in seconds.
if ("--check" %in% WANT) {
  clin   <- file.path(root, "data", "clinical_data.xlsx")
  paired <- file.path(root, "output", "paired_deconv.rds")
  arm_detail <- paste0("arm ", ARM, " cells at ", as_spec$region_csv$path,
                       if (have_arm) "" else " ABSENT")
  row <- function(fig, ok, detail)
    data.frame(figure = fig, status = if (ok) "READY" else "MISSING", detail = detail)
  chk <- rbind(
    row("S3b", have_arm, arm_detail),
    row("S9",  have_arm, arm_detail),
    row("S10", have_arm && file.exists(clin),
        paste0(arm_detail, "; ", clin, if (file.exists(clin)) "" else " ABSENT",
               " (category source ", HOTCOLD_SOURCE, ")")),
    row("S11", file.exists(paired),
        paste0(paired, if (file.exists(paired)) "" else
          " ABSENT: knit analysis/molecular_massimo2.Rmd (IHC_KNIT_MOLECULAR=1)")))
  dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(chk, file.path(OUT, "check_ihc.csv"), row.names = FALSE)
  print(chk, right = FALSE)
  quit(save = "no", status = 0)
}
groups   <- NULL
if (have_arm) {
  as_cells  <- arm_cells(as_spec)
  as_ucells <- arm_union_tier_cells(as_spec)
  pids <- unique(c(as_cells$patient_id,
                   if ("patient_id" %in% names(as_ucells)) as_ucells$patient_id))
  as_polys <- tryCatch(arm_annotations(as_spec, "region", patient_ids = pids),
                       error = function(e) NULL)
  as_upoly <- tryCatch(arm_annotations(as_spec, "union", patient_ids = pids),
                       error = function(e) NULL)
  .prom    <- arm_promote_unregioned(as_spec, as_cells, as_polys, as_ucells, as_upoly)
  as_cells <- .prom$cells; as_polys <- .prom$polys
  as_union <- arm_metrics(as_spec, as_cells, as_polys, "union",
                          union_cells = as_ucells, union_polys = as_upoly)
  groups <- paper_hotcold_groups(file.path(root, "data", "clinical_data.xlsx"),
                                 patient_ids = as_union$patient_id,
                                 source = HOTCOLD_SOURCE)
}

# --- S3b: P(CD3+ | CD8+) per case --------------------------------------------------
# A consistency check, not an exclusivity one: CD8 T cells are CD3+, so the fraction
# should sit near 1 in every case. marker_qc.Rmd's exclusivity pairs test the opposite
# claim (co-positivity near 0) and carry no CD3/CD8 row.
if (want("S3")) {
  if (!have_arm) skip("S3b", paste("no cells for arm", ARM)) else {
    # The arm's own cells (FlowPath <M>_sign columns), as every other panel here --
    # not load_data.R's ihc_for(), which drags DESeq2 in for a positivity count.
    pos <- marker_matrix(as_cells, c("CD3", "CD8")) |>
      mutate(patient_id = slide_key(as_cells$patient_id))
    s3b <- pos |> group_by(patient_id) |>
      summarise(n_cd8 = sum(CD8), n_cd8_cd3 = sum(CD8 & CD3),
                frac_cd3_given_cd8 = ifelse(n_cd8 > 0, n_cd8_cd3 / n_cd8, NA_real_),
                .groups = "drop") |>
      arrange(desc(frac_cd3_given_cd8))
    write_values(s3b, "S3", "S3b_values")
    p <- ggplot(s3b, aes(reorder(patient_id, -frac_cd3_given_cd8), frac_cd3_given_cd8)) +
      geom_col(fill = "grey70", width = 0.7) +
      geom_text(aes(label = paste0("n=", n_cd8)), vjust = -0.3, size = 2.2) +
      scale_y_continuous(limits = c(0, 1.08), breaks = seq(0, 1, 0.25)) +
      labs(x = "case", y = "CD3+ among CD8+ cells (fraction)",
           subtitle = paste0("n = ", nrow(s3b), " cases; n above bars = CD8+ cells")) +
      theme_minimal(base_size = BASE_PT) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    save_supp(p, "S3", "S3b_cd3_given_cd8", MM[["one_col"]], 70)
  }
}

# --- S9: every case, cells coloured by the FlowPath phenotype call -----------------
if (want("S9")) {
  if (!have_arm) skip("S9", paste("no cells for arm", ARM)) else {
    cases <- sort(unique(as_cells$patient_id))
    for (cb in c("lineage", "compartment")) {
      for (pid in cases) {
        p <- tryCatch(
          paper_phenotype_map(as_cells, patient_id = pid, annots = as_polys,
                              colour_by = cb, title = pid),
          error = function(e) { message("  S9 ", pid, ": ", conditionMessage(e)); NULL })
        if (!is.null(p)) save_supp(p, "S9", paste0("S9_", cb, "_", pid), 90, 90)
      }
    }
    write_values(tibble(case = cases,
                        group = if (!is.null(groups))
                          groups$group[match(cases, groups$patient_id)] else NA),
                 "S9")
  }
}

# --- S10: CD45+ fraction per case, individually and by transcriptomic category -----
if (want("S10")) {
  if (!have_arm || is.null(groups)) {
    skip("S10", "needs the arm cells and data/clinical_data.xlsx (Immuno-phenotype)")
  } else {
    d <- as_union |> select(patient_id, cd45_over_inside) |>
      left_join(groups, by = "patient_id") |> filter(!is.na(group))
    write_values(d, "S10")
    lv <- intersect(c("cold", "intermediate", "hot"), unique(tolower(d$group)))
    d$group <- factor(tolower(d$group), levels = lv)
    p_each <- ggplot(d, aes(reorder(patient_id, cd45_over_inside), cd45_over_inside,
                            fill = group)) +
      geom_col(width = 0.7) +
      labs(x = "case", y = "mIF CD45+ / all cells", fill = NULL,
           subtitle = paste0("n = ", nrow(d), " cases")) +
      theme_minimal(base_size = BASE_PT) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    p_grp <- paper_immune_fraction_hotcold(as_union, groups)
    save_supp(p_each, "S10", "S10_per_case", MM[["one_half"]], 70)
    save_supp(p_grp, "S10", "S10_grouped", MM[["one_col"]], 70)
    if (requireNamespace("patchwork", quietly = TRUE))
      save_supp(patchwork::wrap_plots(p_each, p_grp, widths = c(2, 1)), "S10",
                "S10_combined", MM[["two_col"]], 75)
  }
}

# --- S11: imaging vs every deconvolution method, raw points, no fitted line --------
if (want("S11")) {
  paired_path <- file.path(root, "output", "paired_deconv.rds")
  if (!file.exists(paired_path)) {
    skip("S11", "output/paired_deconv.rds absent: knit analysis/molecular_massimo2.Rmd")
  } else {
    paired <- readRDS(paired_path)
    write_values(paired, "S11")
    methods <- sort(unique(paired$method))
    plots <- lapply(methods, function(m) tryCatch(
      paper_deconv_scatter(paired, method = m, groups = groups) + ggtitle(m),
      error = function(e) NULL))
    names(plots) <- methods
    ok <- Filter(Negate(is.null), plots)
    for (m in names(ok)) save_supp(ok[[m]], "S11", paste0("S11_", m), MM[["one_half"]], 90)
    if (length(ok) && requireNamespace("patchwork", quietly = TRUE))
      save_supp(patchwork::wrap_plots(ok, ncol = 2), "S11", "S11_all_methods",
                MM[["two_col"]], min(MAX_H, 70 * ceiling(length(ok) / 2)))
  }
}
message("supplementary.R done -> ", OUT)
