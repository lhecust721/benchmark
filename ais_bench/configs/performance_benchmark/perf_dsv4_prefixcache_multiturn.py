# DeepSeek-V4-Flash (w4a8) PD-disaggregated multi-turn prefix-cache performance test.
# 50 ShareGPT conversations, 3-6 turns each, infer_mode="every" (each turn's real
# model output is appended to the next request to exercise Decode KV save and
# Prefill KV-pool reuse through the PD proxy).

from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.sharegpt.sharegpt_prefixcache_gen import (
        sharegpt_datasets as datasets,
    )
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat_multiturn import (
        models,
    )

datasets[0]["path"] = (
    "/workspace/benchmark/ais_bench/datasets/sharegpt/"
    "ShareGPT_prefixcache_16384.json"
)

models[0]["abbr"] = "dsv4-w4a8-prefixcache"
models[0]["path"] = "/data1/weight/YDJaken--DeepSeek-V4-Flash-0731-w4a8"
models[0]["model"] = "dsv4"
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8700
models[0]["stream"] = True
models[0]["request_rate"] = 0
models[0]["retry"] = 2
models[0]["batch_size"] = 16
models[0]["max_out_len"] = 4096
models[0]["generation_kwargs"] = dict(
    temperature=0.01,
    # KV-divergence 修复: 上一轮 ignore_eos=True 时, 模型自然回答中位仅 ~608 token 即出 EOS,
    # 强制续写产生 [. EOS . EOS 8b8b...] 垃圾段, detokenize→re-tokenize 在 EOS 处断裂
    # (实测 98/98 分叉点 = 首个 EOS 位置)。改为 False: 自然收尾含 EOS, round-trip 干净;
    # 自然写满 4096 被截断(finish=length)同样干净。长度不足 4096 的轮次为结构性 0 命中。
    ignore_eos=False,
    min_tokens=4096,  # 强制至少 4096 token 输出(EOS 在此之前被抑制, 不进入序列→无分叉;与 ignore_eos 不同)
    # 思考模式跨轮 KV 复用三条件 (背景 §4b / dsv4-pd-kv-reuse-analysis-background.md §7):
    # thinking=true + drop_thinking=false + assistant 消息带 reasoning 字段。
    # drop_thinking=true (默认) 只渲染 content → 分叉点在 <｜Assistant｜> 后 → 命中 0。
    # 回填侧: icl_multiturn_inferencer.infer_every 已改为同时回填 content 和 reasoning。
    chat_template_kwargs=dict(thinking=True, drop_thinking=False),
    # KV-divergence diagnostic: capture exact token ids for offline replay analysis.
    return_token_ids=True,
)

work_dir = "outputs/perf_dsv4_prefixcache_multiturn/"
