library(arrow)

basePath = './runs/2026-09-29T144956097078Z_b5845b4/'

benchmarkResults = as.data.frame(arrow::read_parquet(file.path(basePath, 'benchmark_results.parquet')))
codecResults = as.data.frame(arrow::read_parquet(file.path(basePath, 'codec_results.parquet')))
errors = as.data.frame(arrow::read_parquet(file.path(basePath, 'errors.parquet')))
images = as.data.frame(arrow::read_parquet(file.path(basePath, 'images.parquet')))
pixelResults = as.data.frame(arrow::read_parquet(file.path(basePath, 'pixel_results.parquet')))
preprocessingResults = as.data.frame(arrow::read_parquet(file.path(basePath, 'preprocessing_results.parquet')))
roiMatches = as.data.frame(arrow::read_parquet(file.path(basePath, 'roi_matches.parquet')))
segmentationResults = as.data.frame(arrow::read_parquet(file.path(basePath, 'segmentation_results.parquet')))
systemBenchmarkResults = as.data.frame(arrow::read_parquet(file.path(basePath, 'system_benchmark_results.parquet')))


## Number of ROIs by treatment
segmentationCounts = aggregate.data.frame(segmentationResults[,c('object_count')],
                               by = segmentationResults[,c('preprocessing_id', 'codec', 'setting_id', 'segmentation_method')],
                               FUN = sum)

{
  par(plt = c(0.25, 0.95, 0.15, 0.9))
  lab = segmentationCounts$setting_id[segmentationCounts$segmentation_method == 'fixed']
  plot(segmentationCounts$x[segmentationCounts$segmentation_method == 'fixed'], 1:length(lab),
       yaxt = 'n',
       ylab = '',
       xlab = 'Total ROI Count')
  axis(2, at = 1:length(lab), labels = lab, las = 2, col = 'white')
  axis(2, at = 0:length(lab)+0.5, labels = NA, las = 2, col = 'black')
  for (i in seq(0, length(lab), by = 2)) {
    rect(0, i+0.5, 50000, i+1.5, col = '#00000015', border = NA)
  }
  mtext('ROIs Identified: Fixed Threshold.', side = 3, adj = 0)
}

{
  par(plt = c(0.25, 0.95, 0.15, 0.9))
  lab = segmentationCounts$setting_id[segmentationCounts$segmentation_method == 'otsu']
  plot(segmentationCounts$x[segmentationCounts$segmentation_method == 'otsu'], 1:length(lab),
       yaxt = 'n',
       ylab = '',
       xlab = 'Total ROI Count')
  axis(2, at = 1:length(lab), labels = lab, las = 2, col = 'white')
  axis(2, at = 0:length(lab)+0.5, labels = NA, las = 2, col = 'black')
  for (i in seq(0, length(lab), by = 2)) {
    rect(0, i+0.5, 50000, i+1.5, col = '#00000015', border = NA)
  }
  mtext('ROIs Identified: OTSU Threshold.', side = 3, adj = 0)
}



benchmarkCounts = aggregate(benchmarkResults[,c('megapixels_per_second')],
                             benchmarkResults[,c('preprocessing_id', 'codec', 'setting_id', 'benchmark_kind', 'operation')],
                             mean)

codecCounts = aggregate(codecResults[,c('bits_per_pixel')],
                        codecResults[,c('preprocessing_id', 'codec', 'setting_id')],
                            mean)


{
  plot(NULL, NULL, xlim = c(0, 8), ylim = c(-5, 5), yaxt = 'n', ylab = '', xlab = 'Bits per Pixel')
  
  for (i in 1:nrow(codecCounts)) {
    if (codecCounts$preprocessing_id[i] == 'original') {
      k = which(benchmarkCounts$preprocessing_id == codecCounts$preprocessing_id[i] &
                  benchmarkCounts$codec == codecCounts$codec[i] &
                  benchmarkCounts$setting_id == codecCounts$setting_id[i] &
                  benchmarkCounts$benchmark_kind == 'in_memory')
      scale = (as.numeric(benchmarkCounts$operation[k] == 'encode')-0.5)*2
      lines(x = rep(codecCounts$x[i], 2), y = log10(benchmarkCounts$x[k]) * scale)
    }
  }
  abline(h = 0)
  
  axis(2, at = c(1:10), labels = 10^(1:10), las = 2)
  axis(2, at = log10(c(1:10, 1:10 * 10, 1:10 * 100, 1:10 * 1e3, 1:10 * 1e4)), labels = NA)
  abline(h = log10(c(1:10, 1:10 * 10, 1:10 * 100, 1:10 * 1e3, 1:10 * 1e4)), lty = 3, col = 'lightgrey')
  
  axis(2, at = -c(1:10), labels = 10^(1:10), las = 2)
  axis(2, at = -log10(c(1:10, 1:10 * 10, 1:10 * 100, 1:10 * 1e3, 1:10 * 1e4)), labels = NA)
  abline(h = -log10(c(1:10, 1:10 * 10, 1:10 * 100, 1:10 * 1e3, 1:10 * 1e4)), lty = 3, col = 'lightgrey')
  
  mtext('Encoding\nThroughput\n(MP/s)', 2, adj = 0.75, line = 4)
  mtext('Decoding\nThroughput\n(MP/s)', 2, adj = 0.25, line = 4)
}


{
  plot(NULL, NULL, xlim = c(0, 8), ylim = c(0, 5), yaxt = 'n', ylab = '', xlab = 'Bits per Pixel')
  
  for (i in 1:nrow(codecCounts)) {
    if (codecCounts$preprocessing_id[i] == 'original') {
      k = which(benchmarkCounts$preprocessing_id == codecCounts$preprocessing_id[i] &
                  benchmarkCounts$codec == codecCounts$codec[i] &
                  benchmarkCounts$setting_id == codecCounts$setting_id[i] &
                  benchmarkCounts$benchmark_kind == 'roundtrip' & benchmarkCounts$operation =='decode')
      points(x = codecCounts$x[i], y = log10(benchmarkCounts$x[k]))
    }
  }
  abline(h = 0)
  
  axis(2, at = c(1:10), labels = 10^(1:10), las = 2)
  axis(2, at = log10(c(1:10, 1:10 * 10, 1:10 * 100, 1:10 * 1e3, 1:10 * 1e4)), labels = NA)
  abline(h = log10(c(1:10, 1:10 * 10, 1:10 * 100, 1:10 * 1e3, 1:10 * 1e4)), lty = 3, col = 'lightgrey')
  
  mtext('Roundtrip\nThroughput\n(MP/s)', 2, adj = 0.5, line = 4)
}





roiCounts = aggregate(roiMatches[,c('iou', 'relative_area_error', 'reference_area')],
          roiMatches[,c('codec', 'setting_id', 'segmentation_method', 'relationship')],
          mean)

{
  k = which(roiCounts$segmentation_method == 'fixed' & roiCounts$relationship == '1:1')
  
  lab = roiCounts$setting_id[k]
  plot(roiCounts$iou[k], c(1:length(lab)), yaxt = 'n', ylab = '', xlab = 'IOU')
  
  axis(2, at = 1:length(lab), labels = lab, las = 2, col = 'white')
  axis(2, at = 0:length(lab)+0.5, labels = NA, las = 2, col = 'black')
  for (i in seq(0, length(lab), by = 2)) {
    rect(0, i+0.5, 50000, i+1.5, col = '#00000015', border = NA)
  }
  mtext('IOU of Matching ROIs: Fixed Threshold.', side = 3, adj = 0)
}

{
  k = which(roiCounts$segmentation_method == 'otsu' & roiCounts$relationship == '1:1')
  
  lab = roiCounts$setting_id[k]
  plot(roiCounts$iou[k], c(1:length(lab)), yaxt = 'n', ylab = '', xlab = 'IOU')
  
  axis(2, at = 1:length(lab), labels = lab, las = 2, col = 'white')
  axis(2, at = 0:length(lab)+0.5, labels = NA, las = 2, col = 'black')
  for (i in seq(0, length(lab), by = 2)) {
    rect(0, i+0.5, 50000, i+1.5, col = '#00000015', border = NA)
  }
  mtext('IOU of Matching ROIs: OTSU Threshold.', side = 3, adj = 0)
}


## Plot changes in IOU for a codec by size bins
sizes = c(5:9 * 10, 1:9 * 100, 1:9 * 1000)

plot(NULL, NULL, xlim = range(log10(sizes)), ylim = c(0,1), ylab = 'IOU', yaxs = 'i')
for (i in 1:(length(sizes)-1)) {
  k = which(roiMatches$reference_area >= sizes[i] & roiMatches$reference_area < sizes[i+1])
  if (length(k) > 0) {
    boxplot.default(roiMatches$iou[k], at = log10(sizes[i]), add = T, width = 0.01)
  }
}


addBox = function(x, y) {
  iqr = IQR(y)
}



