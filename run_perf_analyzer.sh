mkdir -p "/app/triton/outputs/batched_full";
for batch_size in 1 2 4 8 16 32 64 128 256
do
  perf_analyzer -m tts_hi_batched --input-data=/app/triton/input-$batch_size.json -b1 --concurrency-range 1:1 --collect-metrics --verbose-csv -f /app/triton/outputs/batched_full/output-b$batch_size.csv
done