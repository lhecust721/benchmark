from ais_bench.benchmark.datasets import ShareGPTDataset, ShareGPTEvaluator
from ais_bench.benchmark.openicl.icl_inferencer import MultiTurnGenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import MultiTurnPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever


sharegpt_reader_cfg = dict(
    input_columns=["question", "answer"],
    output_column="answer",
)


sharegpt_infer_cfg = dict(
    prompt_template=dict(
        type=MultiTurnPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="{question}"),
                dict(role="BOT", prompt="{answer}"),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    # "every" appends each real model response to the next request. This is
    # required to validate Decode KV save followed by Prefill KV-pool reuse.
    inferencer=dict(type=MultiTurnGenInferencer, infer_mode="every"),
)


sharegpt_eval_cfg = dict(evaluator=dict(type=ShareGPTEvaluator))


sharegpt_datasets = [
    dict(
        abbr="sharegpt-prefixcache",
        type=ShareGPTDataset,
        disable_shuffle=True,
        path=(
            "ais_bench/datasets/sharegpt/"
            "ShareGPT_prefixcache_256.json"
        ),
        reader_cfg=sharegpt_reader_cfg,
        infer_cfg=sharegpt_infer_cfg,
        eval_cfg=sharegpt_eval_cfg,
    )
]
