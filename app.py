import gradio as gr

from src.generate import generate
from src.retrieve import retrieve


def handle_query(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""
    chunks = retrieve(question)
    answer = generate(question, chunks)
    sources = "\n".join(f"• {c['source_file']}" for c in chunks)
    return answer, sources


with gr.Blocks(title="WMU Housing Guide") as demo:
    gr.Markdown(
        "# WMU Housing Unofficial Guide\n"
        "Ask any question about on-campus or off-campus housing at Western Michigan University."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. Is the KMetro bus free for WMU students?",
    )
    btn = gr.Button("Ask", variant="primary")
    answer_out = gr.Textbox(label="Answer", lines=10)
    sources_out = gr.Textbox(label="Retrieved from", lines=4)
    btn.click(handle_query, inputs=inp, outputs=[answer_out, sources_out])
    inp.submit(handle_query, inputs=inp, outputs=[answer_out, sources_out])


if __name__ == "__main__":
    demo.launch()
