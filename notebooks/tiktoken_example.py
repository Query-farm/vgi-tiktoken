"""Presentation for token accounting; worker computation remains in visible SQL."""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from vgi_notebooks import finish as finish_chart, prose

ENCODING_COLORS = {"cl100k_base": "#456fba", "o200k_base": "#267358"}


def finish(fig, title, subtitle, name):
    finish_chart(
        fig,
        title,
        subtitle,
        "Source: original teaching text · VGI tiktoken v0.4.1 · ordinary-text BPE counts",
        name=name,
    )


def plot_counts(comparison, selected_encoding):
    y = np.arange(len(comparison))
    fig, ax = plt.subplots(figsize=(10.8, 7.8))
    for offset, (encoding, column) in zip(
        (-0.18, 0.18),
        (("cl100k_base", "cl100k_tokens"), ("o200k_base", "o200k_tokens")),
        strict=True,
    ):
        bars = ax.barh(
            y + offset,
            comparison[column],
            height=0.31,
            color=ENCODING_COLORS[encoding],
            edgecolor="#253d31" if encoding == selected_encoding else "none",
            linewidth=1.5,
            label=encoding,
        )
        ax.bar_label(bars, padding=4, fontsize=9)
    ax.set(
        yticks=y,
        yticklabels=comparison.label,
        xlabel="Tokens in the complete input string",
        ylabel="",
    )
    ax.invert_yaxis()
    ax.set_xlim(0, comparison[["cl100k_tokens", "o200k_tokens"]].to_numpy().max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=True)
    finish(
        fig,
        "The text stays fixed; the encoding changes",
        f"Eleven teaching examples · selected: {selected_encoding} (outlined)",
        f"counts-{selected_encoding}",
    )
    fewer = int((comparison.o200k_tokens < comparison.cl100k_tokens).sum())
    equal = int((comparison.o200k_tokens == comparison.cl100k_tokens).sum())
    more = int((comparison.o200k_tokens > comparison.cl100k_tokens).sum())
    prose(
        f"On these **{len(comparison)} fixed strings**, o200k_base uses fewer tokens in **{fewer}**, the same number in **{equal}**, and more in **{more}**. These are per-string comparisons, not a general ranking of encodings."
    )


def plot_ratios(selected, selected_encoding):
    display(selected)
    y = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.5), sharey=True)
    for ax, column, label in zip(
        axes,
        ("code_points_per_token", "bytes_per_token"),
        ("Unicode code points per token", "UTF-8 bytes per token"),
        strict=True,
    ):
        bars = ax.barh(y, selected[column], height=0.62, color=ENCODING_COLORS[selected_encoding])
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        ax.set(xlabel=label, xlim=(0, selected[column].max() * 1.25))
        ax.grid(axis="y", visible=False)
    axes[0].set(yticks=y, yticklabels=selected.label)
    axes[0].invert_yaxis()
    finish(
        fig,
        "A token has no fixed character or byte size",
        f"Selected encoding: {selected_encoding} · independent panel scales",
        f"ratios-{selected_encoding}",
    )
    accent = selected[selected.sample_id.isin([10, 11])].sort_values("sample_id")
    a, b = list(accent.itertuples())
    prose(
        f"The precomposed accent uses **{int(a.code_points)} code points / {int(a.utf8_bytes)} bytes / {int(a.token_count)} tokens**; the combining version uses **{int(b.code_points)} / {int(b.utf8_bytes)} / {int(b.token_count)}**. Their appearance is similar, but their underlying representations differ. Normalize only when that transformation fits your data contract."
    )


def plot_windows(windows, overlap):
    display(windows)
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for row in windows.itertuples():
        i = int(row.chunk_number) - 1
        ax.barh(
            i, row.end_token - row.start_token, left=row.start_token, color="#267358", height=0.45
        )
        ax.text(
            (row.start_token + row.end_token) / 2,
            i,
            f"{int(row.start_token)}–{int(row.end_token)}",
            color="white",
            ha="center",
            va="center",
            weight="bold",
            fontsize=11,
        )
    ax.set_yticks(range(len(windows)), [f"Window {int(n)}" for n in windows.chunk_number])
    ax.invert_yaxis()
    ax.set_xlim(0, 17)
    ax.set_xticks(range(0, 17, 2))
    ax.set_xlabel("Position in the original token-ID list · zero-based, end-exclusive")
    ax.grid(axis="y", visible=False)
    total = int(windows.returned_tokens.sum())
    finish(
        fig,
        f"{total} returned tokens describe 16 source tokens",
        f"Eight-token windows · requested overlap {overlap} · ASCII control verified by token IDs",
        f"windows-overlap-{overlap}",
    )
    prose(
        f"The worker returned **{len(windows)} windows and {total} tokens in total**. "
        f"That is **{total - 16} repeated tokens** beyond this document's 16 source tokens. "
        "Each returned ID sequence matches its original slice on this ASCII fixture. "
        "Removing the declared overlap reconstructs the original IDs; these checks are repeated below."
    )
