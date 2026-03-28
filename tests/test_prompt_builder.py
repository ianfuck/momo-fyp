from backend.prompting.prompt_builder import PromptBuilder, validate_generated_sentence
from backend.types import AudienceFeatures


def test_tracking_prompt_uses_stage_specific_examples():
    builder = PromptBuilder(
        "resource/md/system-persona_tracking.md",
        "resource/md/system-persona_idle.md",
    )
    prompt = builder.build_tracking_prompt(
        sentence_index=5,
        selected_examples=[
            "resource/example/track-example-1.csv",
            "resource/example/track-example-2.csv",
        ],
        audience=AudienceFeatures(top_color="粉色", height_class="short", build_class="slim", distance_class="near"),
        event_summary="揮手",
        reacquired=False,
    )
    assert "第 5 句" in prompt["user"]
    assert "揮手" in prompt["user"]
    assert "粉色" in prompt["user"]
    assert "最近歷史" not in prompt["user"]
    assert "你在揮手？是在跟你的理智告別嗎？真可愛。" in prompt["user"]
    assert "0.6 到 0.7" in prompt["system"]
    assert "揮" in prompt["required_terms"]


def test_tracking_prompt_visual_mode_uses_crop_instruction_instead_of_summary():
    builder = PromptBuilder(
        "resource/md/system-persona_tracking.md",
        "resource/md/system-persona_idle.md",
    )
    prompt = builder.build_tracking_prompt(
        sentence_index=2,
        selected_examples=[
            "resource/example/track-example-1.csv",
            "resource/example/track-example-2.csv",
        ],
        audience=AudienceFeatures(top_color="粉色", height_class="short", build_class="slim", distance_class="near"),
        event_summary="無",
        reacquired=False,
        use_visual_audience=True,
    )
    assert "觀眾特徵摘要:" not in prompt["user"]
    assert "觀眾特徵來源:" in prompt["user"]
    assert "crop 圖" in prompt["user"]
    assert prompt["required_terms"] == []


def test_sentence_validator():
    assert validate_generated_sentence("你好。", 22) == []
    assert validate_generated_sentence("", 22)
    assert validate_generated_sentence("這是一句非常非常非常非常非常長的句子", 5)
    assert validate_generated_sentence("。", 22)
    assert validate_generated_sentence("好。", 22)
