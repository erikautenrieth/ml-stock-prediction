import streamlit as st

from frontend.data_loader import load_model_info


def _fmt_pct(v: float | None) -> str:
    return f"{v:.1%}" if v else "—"


def render_model_card() -> None:
    """Render the active model info card."""
    info = load_model_info()

    with st.container(border=True):
        trained = info["trained"]
        status = "✅ Trained" if trained else "⏳ Not trained yet"

        left, right = st.columns([2, 1])
        left.markdown(
            f"**🤖 {info['model_short']}** "
            f"&nbsp;·&nbsp; {info['horizon']}d horizon "
            f"&nbsp;·&nbsp; {status}"
        )
        if info.get("trained_at"):
            right.markdown(
                f"<div style='text-align:right; color:#888;'>"
                f"Last trained: {info['trained_at']}</div>",
                unsafe_allow_html=True,
            )

        if trained:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", _fmt_pct(info["accuracy"]))
            c2.metric("F1", _fmt_pct(info["f1"]))
            c3.metric("Precision", _fmt_pct(info["precision"]))
            c4.metric("Recall", _fmt_pct(info["recall"]))
            c5.metric("ROC AUC", _fmt_pct(info["roc_auc"]))
            if info.get("n_features"):
                st.caption(f"{info['n_features']} features")
        else:
            st.info(
                "Run `make train` to train the model and see metrics here.",
                icon="💡",
            )
