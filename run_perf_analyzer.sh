mkdir -p "/priyam-workspace/tts/tts-benchmarking/analyze/tts_tensorrt_decomposed/";
for batch_size in 1 2 4 8 16 32 64 128 256
do
  perf_analyzer -m tts_tensorrt_decomposed --input-data=/priyam-workspace/tts/tts-benchmarking/inputs/input-$batch_size.json -b1 --concurrency-range 1:1 --collect-metrics --verbose-csv -f /priyam-workspace/tts/tts-benchmarking/analyze/tts_tensorrt_decomposed/output-b$batch_size.csv
done