import re
import gradio as gr
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Literal, List, Tuple
from fireredtts2.fireredtts2 import FireRedTTS2


# ================================================
#                   FireRedTTS2 Model
# ================================================
# Global model instance
model: FireRedTTS2 = None


def initiate_model(pretrained_dir: str, device="cuda"):
    global model
    if model is None:
        model = FireRedTTS2(
            pretrained_dir=pretrained_dir,
            gen_type="dialogue", # Tetap menggunakan dialogue untuk mendukung kedua mode
            device=device,
        )


# ================================================
#                   Gradio
# ================================================

# i18n
_i18n_key2lang_dict = dict(
    # Title markdown
    title_md_desc="FireRedTTS-2 🔥 Speech Generation",
    # Voice mode radio
    voice_mode_label="Voice Mode",
    voice_model_choice1="Voice Clone",
    voice_model_choice2="Random Voice",
    # Speaker1 Prompt
    spk1_prompt_audio_label="Speaker 1 Prompt Audio",
    spk1_prompt_text_label="Speaker 1 Prompt Text",
    spk1_prompt_text_placeholder="[S1] text of speaker 1 prompt audio.",
    # Speaker2 Prompt
    spk2_prompt_audio_label="Speaker 2 Prompt Audio",
    spk2_prompt_text_label="Speaker 2 Prompt Text",
    spk2_prompt_text_placeholder="[S2] text of speaker 2 prompt audio.",
    # Dialogue input textbox
    dialogue_text_input_label="Dialogue Text Input",
    dialogue_text_input_placeholder="[S1]text[S2]text[S1]text...",
    # Monologue input textbox
    monologue_text_input_label="Monologue Text Input",
    monologue_text_input_placeholder="Enter monologue text here.",
     # Monologue prompt text
    monologue_prompt_text_label="Prompt Text (for Voice Clone)",
    monologue_prompt_text_placeholder="Text of the prompt audio.",
    # Monologue prompt audio
    monologue_prompt_audio_label="Prompt Audio (for Voice Clone)",
    # Generate button
    generate_btn_label="Generate Audio",
    # Generated audio
    generated_audio_label="Generated Audio",
    # Warining1: invalid text for prompt
    warn_invalid_spk1_prompt_text='Invalid speaker 1 prompt text, should strictly follow: "[S1]xxx"',
    warn_invalid_spk2_prompt_text='Invalid speaker 2 prompt text, should strictly follow: "[S2]xxx"',
    # Warining2: invalid text for dialogue input
    warn_invalid_dialogue_text='Invalid dialogue input text, should strictly follow: "[S1]xxx[S2]xxx..."',
     # Warning3: incomplete prompt info
    warn_incomplete_prompt="Please provide prompt audio and text for the speakers.",
    # Warning4: empty monologue text
    warn_empty_monologue_text="Monologue text cannot be empty.",
)

def i18n(key):
    return _i18n_key2lang_dict[key]


def check_monologue_text(text: str, prefix: str = None) -> bool:
    text = text.strip()
    # Check speaker tags
    if prefix is not None and (not text.startswith(prefix)):
        return False
    # Remove prefix
    if prefix is not None:
        text = text.removeprefix(prefix)
    text = text.strip()
    # If empty?
    if len(text) == 0:
        return False
    return True


def check_dialogue_text(text_list: List[str]) -> bool:
    if len(text_list) == 0:
        return False
    for text in text_list:
        if not (
            check_monologue_text(text, "[S1]")
            or check_monologue_text(text, "[S2]")
            or check_monologue_text(text, "[S3]")
            or check_monologue_text(text, "[S4]")
        ):
            return False
    return True


def dialogue_synthesis_function(
    target_text: str,
    voice_mode: int,  # 0 means voice clone
    spk1_prompt_text: str | None = "",
    spk1_prompt_audio: str | None = None,
    spk2_prompt_text: str | None = "",
    spk2_prompt_audio: str | None = None,
):
    # Voice clone mode, check prompt info
    if voice_mode == 0: # Voice Clone
        prompt_has_value = [
            spk1_prompt_text != "",
            spk1_prompt_audio is not None,
            spk2_prompt_text != "",
            spk2_prompt_audio is not None,
        ]
        if not all(prompt_has_value):
            gr.Warning(message=i18n("warn_incomplete_prompt"))
            return None
        if not check_monologue_text(spk1_prompt_text, "[S1]"):
            gr.Warning(message=i18n("warn_invalid_spk1_prompt_text"))
            return None
        if not check_monologue_text(spk2_prompt_text, "[S2]"):
            gr.Warning(message=i18n("warn_invalid_spk2_prompt_text"))
            return None

    # Check dialogue text
    target_text_list: List[str] = re.findall(r"(\[S[0-9]\][^\[\]]*)", target_text)
    target_text_list = [text.strip() for text in target_text_list]
    if not check_dialogue_text(target_text_list):
        gr.Warning(message=i18n("warn_invalid_dialogue_text"))
        return None

    # Go synthesis
    progress_bar = gr.Progress(track_tqdm=True)
    prompt_wav_list = (
        None if voice_mode != 0 else [spk1_prompt_audio, spk2_prompt_audio]
    )
    prompt_text_list = None if voice_mode != 0 else [spk1_prompt_text, spk2_prompt_text]
    target_audio = model.generate_dialogue(
        text_list=target_text_list,
        prompt_wav_list=prompt_wav_list,
        prompt_text_list=prompt_text_list,
        temperature=0.9,
        topk=30,
    )
    return (24000, target_audio.squeeze(0).cpu().numpy())


def monologue_synthesis_function(
    text: str,
    voice_mode: int, # 0 means voice clone
    prompt_audio: str | None = None,
    prompt_text: str | None = "",
):
    if not text.strip():
        gr.Warning(message=i18n("warn_empty_monologue_text"))
        return None

    prompt_wav = None
    if voice_mode == 0: # Voice Clone
        if not prompt_audio or not prompt_text.strip():
            gr.Warning(message=i18n("warn_incomplete_prompt"))
            return None
        prompt_wav = prompt_audio

    # Go synthesis
    progress_bar = gr.Progress(track_tqdm=True)
    target_audio = model.generate_monologue(
        text=text,
        prompt_wav=prompt_wav,
        prompt_text=prompt_text,
        temperature=0.75,
        topk=20,
    )
    return (24000, target_audio.squeeze(0).cpu().numpy())


# UI rendering
def render_interface() -> gr.Blocks:
    with gr.Blocks(title="FireRedTTS-2", theme=gr.themes.Default()) as page:
        # ======================== UI ========================
        # A large title
        title_desc = gr.Markdown(value=f"# {i18n('title_md_desc')}")

        with gr.Tabs():
            # ======================== Dialogue Tab ========================
            with gr.TabItem("Dialogue Generation"):
                with gr.Row():
                    dialogue_voice_mode_choice = gr.Radio(
                        choices=[i18n("voice_model_choice1"), i18n("voice_model_choice2")],
                        value=i18n("voice_model_choice1"),
                        label=i18n("voice_mode_label"),
                        type="index",
                        interactive=True,
                    )
                with gr.Row():
                    with gr.Column(scale=1):
                         with gr.Group(visible=True) as spk1_prompt_group:
                            spk1_prompt_audio = gr.Audio(
                                label=i18n("spk1_prompt_audio_label"),
                                type="filepath",
                                editable=False,
                                interactive=True,
                            )
                            spk1_prompt_text = gr.Textbox(
                                label=i18n("spk1_prompt_text_label"),
                                placeholder=i18n("spk1_prompt_text_placeholder"),
                                lines=3,
                            )
                    with gr.Column(scale=1):
                        with gr.Group(visible=True) as spk2_prompt_group:
                            spk2_prompt_audio = gr.Audio(
                                label=i18n("spk2_prompt_audio_label"),
                                type="filepath",
                                editable=False,
                                interactive=True,
                            )
                            spk2_prompt_text = gr.Textbox(
                                label=i18n("spk2_prompt_text_label"),
                                placeholder=i18n("spk2_prompt_text_placeholder"),
                                lines=3,
                            )
                with gr.Row():
                    dialogue_text_input = gr.Textbox(
                        label=i18n("dialogue_text_input_label"),
                        placeholder=i18n("dialogue_text_input_placeholder"),
                        lines=10,
                    )

                dialogue_generate_btn = gr.Button(
                    value=i18n("generate_btn_label"), variant="primary", size="lg"
                )
                dialogue_generate_audio = gr.Audio(
                    label=i18n("generated_audio_label"),
                    interactive=False,
                )

            # ======================== Monologue Tab ========================
            with gr.TabItem("Monologue Generation"):
                with gr.Row():
                     monologue_voice_mode_choice = gr.Radio(
                        choices=[i18n("voice_model_choice1"), i18n("voice_model_choice2")],
                        value=i18n("voice_model_choice1"),
                        label=i18n("voice_mode_label"),
                        type="index",
                        interactive=True,
                    )
                with gr.Row():
                    with gr.Group(visible=True) as monologue_prompt_group:
                        monologue_prompt_audio = gr.Audio(
                            label=i18n("monologue_prompt_audio_label"),
                            type="filepath",
                            editable=False,
                            interactive=True,
                        )
                        monologue_prompt_text = gr.Textbox(
                            label=i18n("monologue_prompt_text_label"),
                            placeholder=i18n("monologue_prompt_text_placeholder"),
                            lines=3,
                        )
                with gr.Row():
                    monologue_text_input = gr.Textbox(
                        label=i18n("monologue_text_input_label"),
                        placeholder=i18n("monologue_text_input_placeholder"),
                        lines=10,
                    )
                monologue_generate_btn = gr.Button(
                    value=i18n("generate_btn_label"), variant="primary", size="lg"
                )
                monologue_generate_audio = gr.Audio(
                    label=i18n("generated_audio_label"),
                    interactive=False,
                )

        # ======================== Action ========================
        def _change_prompt_input_visibility(voice_mode):
            enable = voice_mode == 0
            return gr.update(visible=enable)

        dialogue_voice_mode_choice.change(
            fn=lambda x: [gr.update(visible=x==0), gr.update(visible=x==0)],
            inputs=[dialogue_voice_mode_choice],
            outputs=[spk1_prompt_group, spk2_prompt_group],
        )

        monologue_voice_mode_choice.change(
            fn=_change_prompt_input_visibility,
            inputs=[monologue_voice_mode_choice],
            outputs=[monologue_prompt_group],
        )

        dialogue_generate_btn.click(
            fn=dialogue_synthesis_function,
            inputs=[
                dialogue_text_input,
                dialogue_voice_mode_choice,
                spk1_prompt_text,
                spk1_prompt_audio,
                spk2_prompt_text,
                spk2_prompt_audio,
            ],
            outputs=[dialogue_generate_audio],
        )

        monologue_generate_btn.click(
            fn=monologue_synthesis_function,
            inputs=[
                monologue_text_input,
                monologue_voice_mode_choice,
                monologue_prompt_audio,
                monologue_prompt_text
            ],
            outputs=[monologue_generate_audio],
        )
    return page


# ================================================
#                   Options
# ================================================
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained-dir", type=str, required=True, help="Directory containing pretrained models.")
    parser.add_argument("--share", action='store_true', help="Create a public link for the Gradio interface.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="The server name to bind to.")
    parser.add_argument("--server-port", type=int, default=7860, help="The port number to launch the server on.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Initiate model
    initiate_model(args.pretrained_dir)
    print("[INFO] FireRedTTS-2 loaded")
    # UI
    page = render_interface()
    page.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)
