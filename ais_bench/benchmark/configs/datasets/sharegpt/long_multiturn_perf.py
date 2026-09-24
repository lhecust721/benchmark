"""Standalone AISBench config. Replace the deployment values below."""
from mmengine.config import read_base
from ais_bench.benchmark.datasets import ShareGPTDataset, ShareGPTEvaluator
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.openicl.icl_inferencer import MultiTurnGenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import MultiTurnPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

models = [dict(
    attr="service",
    type=VLLMCustomAPIChat,
    abbr="long-multiturn-api-chat-stream",
    path="/data/models/YOUR_MODEL",       # Tokenizer path on benchmark client
    model="YOUR_SERVED_MODEL_NAME",      # Model name accepted by the PD proxy
    host_ip="127.0.0.1",                 # PD proxy, not a standalone Decode endpoint
    host_port=8000,
    url="",
    api_key="",
    stream=True,
    request_rate=0,
    retry=0,
    max_out_len=400,
    batch_size=1,
    trust_remote_code=False,
    # Keep normal EOS stopping. A hidden EOS in a continuing decode stream
    # cannot be replayed from AISBench's next-turn assistant text.
    generation_kwargs=dict(temperature=0.0, min_tokens=350, ignore_eos=False),
)]

datasets = [dict(
    abbr="sharegpt-long-10turn",
    type=ShareGPTDataset,
    path="/data/aisbench-long/sharegpt_20k_1k_10turn.json",
    disable_shuffle=True,
    # Deliberately omit max_out_len: placeholder length must not control generation.
    reader_cfg=dict(input_columns=["question", "answer"], output_column="answer"),
    infer_cfg=dict(
        prompt_template=dict(
            type=MultiTurnPromptTemplate,
            template=dict(round=[
                dict(role="HUMAN", prompt="{question}"),
                dict(role="BOT", prompt="{answer}"),
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=MultiTurnGenInferencer, infer_mode="every"),
    ),
    eval_cfg=dict(evaluator=dict(type=ShareGPTEvaluator)),
)]

work_dir = "outputs/long_multiturn"
