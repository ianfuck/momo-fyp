from argparse import Namespace

from scripts import gpt_sovits_v4_fewshot


def test_load_reference_transcript_reads_utf8_file_and_flattens_lines(tmp_path):
    transcript_path = tmp_path / "alien.txt"
    transcript_path.write_text("第一行。\n\n第二行。\r\n第三行。  \n", encoding="utf-8")

    assert gpt_sovits_v4_fewshot._load_reference_transcript(str(transcript_path)) == "第一行。第二行。第三行。"


def test_main_writes_transcript_contents_into_metadata_list(tmp_path, monkeypatch):
    base_model_path = tmp_path / "base-model"
    base_model_path.mkdir()
    audio_path = tmp_path / "alien.mp3"
    audio_path.write_bytes(b"fake-audio")
    transcript_path = tmp_path / "alien.txt"
    transcript_path.write_text("鋼鐵的呼吸，\n齒輪咬合的聲音。", encoding="utf-8")

    monkeypatch.setattr(
        gpt_sovits_v4_fewshot,
        "parse_args",
        lambda: Namespace(
            base_model_path=str(base_model_path),
            reference_audio=str(audio_path),
            reference_transcript=str(transcript_path),
            name="my_gpt_sovits_v4_model-01",
            output_root=str(tmp_path / "output"),
            work_root=str(tmp_path / "work"),
            speaker_name="speaker0",
            language="zh",
            device="cpu",
            batch_size=1,
            s1_epochs=1,
            s2_epochs=1,
            dry_run=True,
        ),
    )
    monkeypatch.setattr(gpt_sovits_v4_fewshot, "_patch_gpt_sovits_frontend", lambda _path: None)
    monkeypatch.setattr(gpt_sovits_v4_fewshot, "_prepare_dataset", lambda *_args, **_kwargs: None)

    assert gpt_sovits_v4_fewshot.main() == 0

    metadata_path = tmp_path / "work" / "my_gpt_sovits_v4_model-01" / "metadata.list"
    assert metadata_path.read_text(encoding="utf-8") == f"{audio_path.resolve()}|speaker0|zh|鋼鐵的呼吸，齒輪咬合的聲音。\n"


def test_normalize_semantic_tsv_adds_header_for_single_sample(tmp_path):
    semantic_path = tmp_path / "6-name2semantic.tsv"
    semantic_path.write_text("alien.mp3\t1 2 3\n", encoding="utf-8")

    gpt_sovits_v4_fewshot._normalize_semantic_tsv(semantic_path)

    assert semantic_path.read_text(encoding="utf-8") == "item_name\tsemantic_audio\nalien.mp3\t1 2 3\n"
