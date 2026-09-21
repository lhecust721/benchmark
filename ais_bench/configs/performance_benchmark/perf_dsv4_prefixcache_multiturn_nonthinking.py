# DeepSeek-V4-Flash (w4a8) PD-disaggregated multi-turn prefix-cache performance test
# -- NON-THINKING mode.
# 基于 perf_dsv4_prefixcache_multiturn.py (思考模式) 修改, 仅切换 thinking 开关:
#   - thinking=False → 模板走 chat 模式 (无 <think>/</think>, 无 reasoning 段)
#   - 不考虑 reasoning 处理: 去掉 drop_thinking 参数 (chat 模式下不渲染 reasoning),
#     ais_bench 的 infer_every 仅在 output.reasoning_content 非空时才回填 reasoning,
#     非思考模式下恒为空, 无需额外处理
#   - 其余配置 (数据集/min_tokens/return_token_ids/batch_size 等) 与思考模式完全一致
#
# 跨轮复用条件 (非思考): assistant content 原样回填即可 (token 级 round-trip, 背景实测
#   96/100 完整匹配, 1-4/100 存在 BPE 重编码分叉, 命中概率性受内容影响)。

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

models[0]["abbr"] = "dsv4-w4a8-prefixcache-nonthink"
models[0]["path"] = "/data1/weight/YDJaken--DeepSeek-V4-Flash-0731-w4a8"
models[0]["model"] = "dsv4"
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8700
models[0]["stream"] = True
models[0]["request_rate"] = 0
models[0]["retry"] = 2
models[0]["batch_size"] = 12
models[0]["max_out_len"] = 4096
models[0]["generation_kwargs"] = dict(
    temperature=0.01,
    # KV-divergence 修复: ignore_eos=False + min_tokens=4096 (自然收尾含 EOS,
    # EOS 不进入序列 → 无分叉), 与思考模式一致
    ignore_eos=False,
    min_tokens=4096,
    # 非思考模式: thinking=False (deepseek_v4 模板缺省为 thinking_enabled=True,
    # 必须显式关闭). chat 模式渲染: <｜User｜>...<｜Assistant｜></think>, assistant1
    # 回填 = content + EOS, 与 decode 保存序列 token 级对齐 (无 reasoning 段)
    chat_template_kwargs=dict(thinking=False),
    # KV-divergence diagnostic: capture exact token ids for offline replay analysis.
    return_token_ids=True,
)

work_dir = "outputs/perf_dsv4_prefixcache_multiturn_nonthink/"
