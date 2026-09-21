# GLM-5.2 (w4a8) standalone 单实例多轮 prefix-cache 分叉验证 -- 非思考模式
# 参考 perf_dsv4_prefixcache_multiturn_nonthinking.py (DSV4 PD 分离版) 适配:
#   - 服务: glm52_standalone.sh 单实例 (8195, TP8, kv_producer + save_decode_cache,
#     consumer_is_to_put; --no-enable-prefix-caching → 跨轮复用只走 AscendStore
#     external 路径, /metrics 看 external_prefix_cache_hits_total)
#   - 数据集: ShareGPT_prefixcache_glm.json (20 会话 × 2-4 轮, 原始 ShareGPT 对话)
#   - 非思考模式: GLM 模板必须显式 enable_thinking=False (缺省为思考模式),
#     渲染 generation prompt 以 <|assistant|><think></think> 结尾,
#     第 2 轮 assistant 区域 = <think></think> + content.strip()
#   - 复用粒度 128 (glm_moe_dsa 单 KV 组, hash_block_size=128; DSV4 为 4096):
#     自然长度生成即可跨越多个 128 粒度边界 → 不需要 DSV4 的 min_tokens=4096
#     强制长生成 (该参数是 DSV4 为对齐 4096 粒度 + 消除 EOS 垃圾分叉的特有配置;
#     GLM 自然停止时 EOS 不进入响应文本, 本身即无 EOS 分叉机制)
#   - return_token_ids=True: 采集服务端渲染 prompt_token_ids + 生成 gen_token_ids
#     (vLLM 0.23.0 流式布局: 首块顶层 prompt_token_ids, 后续块 choices[i].token_ids),
#     供 analyze_prefixcache_divergence.py 离线重放, 与下一轮 prefill 分词结果对比
#
# 验证目标: decode 保存 KV → 下一轮 prefill 复用时是否存在 BPE 分叉
#   decode 保存序列 (turn k)   = prompt_token_ids[k] + gen_token_ids[k] (+EOS 如有)
#   prefill 查询序列 (turn k+1) = prompt_token_ids[k+1] (服务端真实渲染)
#   LCP(保存, 查询) // 128 * 128 = 跨轮可复用 token 数

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
    "ShareGPT_prefixcache_glm.json"
)

models[0]["abbr"] = "glm52-w4a8-prefixcache-nonthink"
models[0]["path"] = "/data1/weight/GLM-5.2-w4a8"
models[0]["model"] = "glm5.2"
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8195
models[0]["stream"] = True
models[0]["request_rate"] = 0
models[0]["retry"] = 2
models[0]["batch_size"] = 20
models[0]["max_out_len"] = 1024
models[0]["generation_kwargs"] = dict(
    temperature=0.01,
    # 自然停止: EOS/stop 不进入响应文本, 下一轮渲染以 <|user|> 续接,
    # token 级 round-trip 无 EOS 分叉机制; 不用 ignore_eos=True (会把 EOS
    # 压进序列产生退化续写, DSV4 已实证其分叉危害)
    ignore_eos=False,
    # 非思考模式: GLM 模板缺省 thinking, 必须显式关闭;
    # 全链 kwargs 一致 (每轮请求都带同一 chat_template_kwargs)
    chat_template_kwargs=dict(enable_thinking=False),
    # KV-divergence diagnostic: capture exact token ids for offline replay analysis.
    return_token_ids=True,
)

work_dir = "/workspace/outputs/perf_glm52_prefixcache_multiturn_nonthink/"
