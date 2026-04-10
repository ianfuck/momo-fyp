from __future__ import annotations

from pathlib import Path

from backend.prompting.examples_loader import load_idle_examples, load_tracking_examples
from backend.types import AudienceFeatures

HEIGHT_LABELS = {"tall": "高個", "medium": "中等身高", "short": "矮個", "unknown": "身高未明"}
BUILD_LABELS = {"broad": "厚實體型", "average": "普通體型", "slim": "偏瘦體型", "unknown": "體型未明"}
DISTANCE_LABELS = {"too_close": "過近", "near": "偏近", "mid": "中距離", "far": "偏遠", "unknown": "距離未明"}


class PromptBuilder:
    def __init__(self, tracking_system_path: str, idle_system_path: str) -> None:
        self.tracking_system = Path(tracking_system_path).read_text(encoding="utf-8")
        self.idle_system = Path(idle_system_path).read_text(encoding="utf-8")

    def build_tracking_prompt(
        self,
        sentence_index: int,
        selected_examples: list[str],
        audience: AudienceFeatures,
        event_summary: str,
        reacquired: bool,
        use_visual_audience: bool = False,
        liberation_mode: bool = False,
    ) -> dict[str, str | list[str]]:
        stages = load_tracking_examples(selected_examples)
        stage_examples = stages.get(sentence_index, [])
        if not stage_examples:
            raise ValueError(f"no stage examples found for sentence {sentence_index}")
        best_reference = self._pick_best_reference(stage_examples, audience, event_summary)
        style_lines = "\n".join(
            f"- 階段靈感{index + 1}: {row['event_hint'] or '一般觀察'}"
            for index, row in enumerate(stage_examples)
        )
        reacquire_lines = ""
        if reacquired and 0 in stages:
            reacquire_lines = "\n".join(
                f"- 重獲時偏向: {row['event_hint'] or '重新追上目標'}" for row in stages[0]
            )
        system_prompt = (
            f"{self.tracking_system.strip()}\n\n"
            "硬性輸出規則:\n"
            "1. 只輸出最終台詞，不要解釋，不要思考過程，不要條列。\n"
            "2. 不要加引號、前綴、角色名、備註。\n"
            "3. 只允許一行繁體中文完整句子。\n"
            "4. 句長必須在 8 到 22 字之間。\n"
            "5. 若未滿足條件，立刻重寫，不可輸出空字串。\n"
            "6. 這次不是自由創作，而是 reference 句的定向改寫。\n"
            "7. 要保留 reference 的句型骨架、停頓位置、前半句與後半句的功能，再把內容替換成當前觀眾特徵與事件。\n"
            "8. 目標是和 reference 保持約 0.6 到 0.7 的相似度：像同一句的改寫版，不可完全照抄，也不可離得太遠。"
        )
        if liberation_mode:
            user_prompt = (
                f"任務: 生成第 {sentence_index} 句追蹤台詞。\n"
                "這次改成解放模式，不做 reference 改寫。\n"
                f"{self._audience_input_line(audience, use_visual_audience)}\n"
                f"即時事件: {event_summary or '無特殊事件'}\n"
                "請緊跟著這張人像 crop 圖與即時事件做反應。\n"
                "口氣要像情緒瞬間爆開，只說一句很少字數、可直接朗讀的話。\n"
                "直接輸出台詞，不要解釋，不要前綴，不要補充說明。"
            )
            return {
                "system": system_prompt,
                "user": user_prompt,
                "required_terms": [],
            }
        user_prompt = (
            f"任務: 生成第 {sentence_index} 句追蹤台詞。\n"
            f"階段限制: 只能參考第 {sentence_index} 句 examples 的句法骨架、壓迫程度、觀察方向與節奏，不可跳段。\n"
            f"本次最貼近的 reference 句:\n- 句子: {best_reference['example_text']}\n"
            f"- 為何選它: {best_reference['event_hint'] or '同一句階段'}\n"
            f"第 {sentence_index} 句階段靈感:\n{style_lines}\n"
            f"{reacquire_lines}\n"
            f"{self._audience_input_line(audience, use_visual_audience)}\n"
            f"即時事件: {event_summary or '無特殊事件'}\n"
            f"本句觀察優先順序: {self._priority_hint(event_summary, audience, use_visual_audience)}\n"
            "改寫方法:\n"
            "- 先把 reference 句當底稿\n"
            "- 只替換其中的顏色、體型、動作、距離、身體部位或形容詞\n"
            "- 其餘句法、停頓、問句或陳述句形式盡量保留\n"
            "- 讓人一眼看出這是同一句的改寫版\n"
            "輸出檢查:\n"
            "- 必須反映至少一個觀眾特徵或事件\n"
            "- 若有即時事件，台詞核心必須先落在事件上\n"
            "- 若沒有即時事件，台詞核心要落在顏色、距離、身形三者之一\n"
            "- 禁止只寫抽象感受而不點出具體觀察\n"
            "- 要像 reference 的同一句改寫，不可完全換成另一種寫法\n"
            "- 儘量保留 reference 的停頓位置、前後半句結構、問句或陳述句型\n"
            "- 允許替換 reference 中的顏色、體型、動作、距離詞\n"
            "- 必須是完整句子\n"
            "- 必須能直接拿去朗讀\n"
            "- 直接輸出台詞，不要任何其他文字"
        )
        return {
            "system": system_prompt,
            "user": user_prompt,
            "required_terms": self._required_terms(event_summary, audience, use_visual_audience),
        }

    def build_idle_prompt(self, selected_examples: list[str], idle_duration_ms: int) -> dict[str, str | list[str]]:
        examples = load_idle_examples(selected_examples)
        examples_text = "\n".join(
            f"- 例句: {row['語音文本內容']} | 氛圍:{row['氛圍提示']}" for row in examples[:10]
        )
        user_prompt = (
            f"目前閒置時間: {idle_duration_ms} ms\n"
            f"{examples_text}\n"
            "輸出要求: 單句、繁體中文、15字內。"
        )
        return {"system": self.idle_system, "user": user_prompt, "required_terms": []}

    def _summarize_audience(self, audience: AudienceFeatures) -> str:
        parts = [
            f"上衣{audience.top_color}",
            f"下身{audience.bottom_color}",
            HEIGHT_LABELS.get(audience.height_class, audience.height_class),
            BUILD_LABELS.get(audience.build_class, audience.build_class),
            f"距離{DISTANCE_LABELS.get(audience.distance_class, audience.distance_class)}",
        ]
        if audience.eye_confidence:
            parts.append(f"眼部追視可信度{audience.eye_confidence:.2f}")
        if audience.focus_score:
            parts.append(f"清晰度{audience.focus_score:.2f}")
        return "、".join(parts)

    def _audience_input_line(self, audience: AudienceFeatures, use_visual_audience: bool) -> str:
        if use_visual_audience:
            return "觀眾特徵來源: 請直接根據提供的人像 crop 圖，自行判讀觀眾的顏色、體型、距離、姿態與可見身體線索，再完成改寫。"
        return f"觀眾特徵摘要: {self._summarize_audience(audience)}"

    def _priority_hint(self, event_summary: str, audience: AudienceFeatures, use_visual_audience: bool = False) -> str:
        if event_summary and event_summary != "無":
            return f"先寫事件「{event_summary}」，再補一個外觀或距離特徵"
        if use_visual_audience:
            return "優先從 crop 圖判斷距離感，再從上衣顏色或體型擇一補充"
        return (
            f"優先寫距離{DISTANCE_LABELS.get(audience.distance_class, audience.distance_class)}，"
            f"再從上衣{audience.top_color}或{BUILD_LABELS.get(audience.build_class, audience.build_class)}擇一補充"
        )

    def _required_terms(self, event_summary: str, audience: AudienceFeatures, use_visual_audience: bool = False) -> list[str]:
        if event_summary and event_summary != "無":
            keywords: list[str] = []
            if "揮手" in event_summary:
                keywords.extend(["揮", "手"])
            if "蹲" in event_summary:
                keywords.append("蹲")
            if "失焦" in event_summary or "模糊" in event_summary:
                keywords.extend(["失焦", "模糊", "近"])
            if "貼近" in event_summary or "太近" in event_summary:
                keywords.extend(["近", "貼"])
            if "遠離" in event_summary:
                keywords.extend(["遠", "退"])
            return list(dict.fromkeys(keywords))
        if use_visual_audience:
            return []
        return [audience.top_color, DISTANCE_LABELS.get(audience.distance_class, audience.distance_class)]

    def _pick_best_reference(
        self,
        stage_examples: list[dict[str, str]],
        audience: AudienceFeatures,
        event_summary: str,
    ) -> dict[str, str]:
        def score(row: dict[str, str]) -> int:
            if row.get("example_text", "").startswith(("(", "（")):
                return -100
            hint = row.get("event_hint", "")
            points = 0
            if event_summary and event_summary != "無":
                for token in ["揮手", "蹲", "失焦", "近", "遠", "粉", "黑", "白", "高", "矮", "胖", "瘦"]:
                    if token in event_summary and token in hint:
                        points += 3
            for token in [audience.top_color, audience.bottom_color]:
                if token != "unknown" and token in hint:
                    points += 2
            shape_tokens = {
                "tall": "高",
                "short": "矮",
                "broad": "胖",
                "slim": "瘦",
            }
            for key, token in shape_tokens.items():
                value = audience.height_class if key in {"tall", "short"} else audience.build_class
                if value == key and token in hint:
                    points += 2
            return points

        return max(stage_examples, key=score)


def validate_generated_sentence(text: str, limit: int) -> list[str]:
    errors: list[str] = []
    cleaned = text.strip().replace("「", "").replace("」", "")
    core = cleaned.strip(" ,.，。!?！？:：;；\"'")
    if "\n" in cleaned:
        errors.append("must be a single line")
    if len(cleaned) > limit:
        errors.append(f"must be <= {limit} chars")
    if not cleaned:
        errors.append("must not be empty")
    chinese_char_count = sum(1 for char in core if "\u4e00" <= char <= "\u9fff")
    if core and chinese_char_count < 2:
        errors.append("must contain at least 2 chinese chars")
    if core and not any("\u4e00" <= char <= "\u9fff" for char in core):
        errors.append("must contain chinese text")
    if cleaned and not core:
        errors.append("must not be punctuation only")
    return errors
