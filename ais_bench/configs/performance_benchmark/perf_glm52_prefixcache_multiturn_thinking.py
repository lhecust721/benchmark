# GLM-5.2 (w4a8) standalone 单实例多轮 prefix-cache 分叉验证 -- 思考模式
# 参考 perf_dsv4_prefixcache_multiturn.py (DSV4 PD 分离思考模式版) 适配:
#   - 服务: glm52_standalone.sh 单实例 (8195, TP8, kv_producer + save_decode_cache,
#     consumer_is_to_put; --no-enable-prefix-caching → 跨轮复用只走 AscendStore
#     external 路径, /metrics 看 external_prefix_cache_hits_total)
#   - 数据集: ShareGPT_prefixcache_glm.json (20 会话 × 2-4 轮)
#   - 复用粒度 128 (glm_moe_dsa 单 KV 组, hash_block_size=128; DSV4 为 4096),
#     通过按轮 min_tokens 保证每轮保存序列跨越多个 128 粒度边界
#   - return_token_ids=True: 采集服务端渲染 prompt_token_ids + 生成 gen_token_ids
#     (vLLM 0.23.0 流式布局: 首块顶层 prompt_token_ids, 后续块 choices[i].token_ids),
#     供 analyze_prefixcache_divergence.py 离线重放, 与下一轮 prefill 分词结果对比
#
# 思考模式跨轮 KV 复用三条件 (chat_template.jinja + 背景 §0.6, 缺一不可):
#   1) enable_thinking=True: 渲染 generation prompt 为 <|assistant|><think> (缺省即思考,
#      但必须显式声明保证全程一致)
#   2) clear_thinking=False: 历史 assistant 消息渲染 <think>{reasoning}</think> 而非
#      空 <think></think> → 与 decode 保存序列 token 级对齐
#   3) assistant 消息带 reasoning 字段: 回填链路已就绪 --
#      vllm_custom_api_chat 捕获 delta.reasoning(_content) → infer_every 回填
#      chat[i]["reasoning"] → _role2api_role 透传 → 服务端 chat_utils 归一化双写
#   另: reasoning_effort 必须全程一致 ("Reasoning Effort" 系统块在第 8 token,
#   Max/High 不一致 → prompt 头部即分叉, 实测教训); high 控制思考长度防超时
#
# 统一每轮最少输出序列长度 (粒度 128 的对齐手段):
#   min_tokens 为标量, 所有轮次相同 (vLLM 原生支持, ais_bench 原样透传);
#   min_tokens == max_out_len 时, EOS/stop 全程被抑制, 每轮确定性写满
#   max_out_len token (finish=length), 保存序列 = prompt + max_out_len,
#   无 EOS 分叉机制

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

models[0]["abbr"] = "glm52-w4a8-prefixcache-think"
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
    # 自然停止/写满截断均无 EOS 分叉机制 (EOS 不进入序列); 不用 ignore_eos=True
    ignore_eos=False,
    # 统一每轮最少输出序列长度 (所有轮相同): == max_out_len →
    # 每轮确定性输出 1024 token, 保存序列必跨 8 个 128 粒度块
    min_tokens=1024,
    # 思考模式: 三条件 + reasoning_effort 全程一致 (见文件头说明)
    chat_template_kwargs=dict(
        enable_thinking=True,
        clear_thinking=False,
        reasoning_effort="high",
    ),
    # KV-divergence diagnostic: capture exact token ids for offline replay analysis.
    return_token_ids=True,
)

work_dir = "outputs/perf_glm52_prefixcache_multiturn_think/"
