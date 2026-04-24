from __future__ import annotations
"""Renders the grounding context and source attribution as HTML for the Streamlit UI."""

import re
import html as html_lib


def build_answer_html(answer_text: str, results: list) -> str:
    """Return a full HTML document with the answer, clickable [N] citations, and collapsible source cards."""

    # Build source data
    sources = []
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        display_name = meta['filename'].replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        sources.append({
            'num': i,
            'filename': meta['filename'],
            'display_name': display_name,
            'page': meta['page_number'],
            'text': html_lib.escape(result['document'])
        })

    safe_answer = html_lib.escape(answer_text)

    # Strip bold markers the LLM sometimes produces
    safe_answer = re.sub(r'\*\*(.+?)\*\*', r'\1', safe_answer)

    # Numbered lists — use temp tag to keep separate from bullet <li> items
    safe_answer = re.sub(r'^(\d+)\. (.+)$', r'<nli>\2</nli>', safe_answer, flags=re.MULTILINE)

    # Bullet points
    safe_answer = re.sub(r'^- (.+)$', r'<li>\1</li>', safe_answer, flags=re.MULTILINE)
    safe_answer = re.sub(r'^\* (.+)$', r'<li>\1</li>', safe_answer, flags=re.MULTILINE)
    safe_answer = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', safe_answer)

    # Wrap numbered list groups in <ol>, convert temp tag to <li>
    def _wrap_ol(m):
        inner = m.group(1).replace('<nli>', '<li>').replace('</nli>', '</li>')
        return f'<ol>{inner}</ol>'
    safe_answer = re.sub(r'((?:<nli>.*?</nli>\n?)+)', _wrap_ol, safe_answer)

    # [N] -> clickable citation pills
    def replace_citation(match):
        num = match.group(1)
        return f'<span class="cite" onclick="showSource({num})">{num}</span>'

    safe_answer = re.sub(r'\[(\d+)\]', replace_citation, safe_answer)

    # Newlines -> paragraphs
    paragraphs = safe_answer.split('\n')
    formatted = []
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith(('<ul>', '<ol>', '<li>', '</ul>', '</ol>')):
            formatted.append(f'<p>{p}</p>')
        elif p:
            formatted.append(p)
    safe_answer = '\n'.join(formatted)

    # Source cards HTML
    source_cards_html = ""
    for src in sources:
        source_cards_html += f"""
        <div class="source-card" id="source-{src['num']}">
            <div class="source-header" onclick="toggleSource({src['num']})">
                <div class="source-num">{src['num']}</div>
                <div class="source-meta">
                    <div class="source-title">{src['display_name']}</div>
                    <div class="source-page">Page {src['page']}</div>
                </div>
                <div class="source-chevron" id="chevron-{src['num']}">
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
            </div>
            <div class="source-body" id="body-{src['num']}">
                <div class="source-text">{src['text']}</div>
                <div class="source-file">{src['filename']}</div>
            </div>
        </div>
        """

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            color: #222;
            line-height: 1.7;
            font-size: 14.5px;
            background: transparent;
            -webkit-font-smoothing: antialiased;
        }}

        .answer-content {{
            padding: 0 0 20px 0;
        }}
        .answer-content p {{
            margin-bottom: 8px;
        }}
        .answer-content ul, .answer-content ol {{
            margin: 6px 0 10px 20px;
            padding: 0;
        }}
        .answer-content li {{
            margin-bottom: 5px;
            line-height: 1.6;
        }}

        .cite {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 600;
            min-width: 15px;
            height: 15px;
            padding: 0 3px;
            border-radius: 3px;
            background: #eee;
            color: #666;
            cursor: pointer;
            vertical-align: super;
            margin: 0 1px;
            line-height: 1;
            transition: all 0.15s ease;
            position: relative;
            top: -1px;
        }}
        .cite:hover {{
            background: #ddd;
            color: #333;
        }}
        .cite.active {{
            background: #333;
            color: #fff;
        }}

        .sources-section {{
            border-top: 1px solid #eee;
            padding-top: 16px;
            margin-top: 4px;
        }}
        .sources-label {{
            font-size: 11px;
            font-weight: 500;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 10px;
        }}

        .source-card {{
            border: 1px solid #eee;
            border-radius: 8px;
            margin-bottom: 6px;
            overflow: hidden;
            transition: border-color 0.2s ease;
            background: #fff;
        }}
        .source-card:hover {{
            border-color: #ccc;
        }}
        .source-card.highlighted {{
            border-color: #333;
        }}

        .source-header {{
            display: flex;
            align-items: center;
            padding: 10px 14px;
            cursor: pointer;
            user-select: none;
            gap: 10px;
        }}
        .source-header:hover {{
            background: #fafafa;
        }}

        .source-num {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 4px;
            background: #f5f5f5;
            color: #666;
            font-size: 11px;
            font-weight: 500;
            flex-shrink: 0;
        }}

        .source-meta {{
            flex: 1;
            min-width: 0;
        }}
        .source-title {{
            font-size: 13px;
            font-weight: 400;
            color: #333;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .source-page {{
            font-size: 11px;
            color: #999;
            margin-top: 1px;
        }}

        .source-chevron {{
            color: #999;
            transition: transform 0.2s ease;
            flex-shrink: 0;
        }}
        .source-chevron.open {{
            transform: rotate(180deg);
        }}

        .source-body {{
            display: none;
            padding: 0 14px 12px 44px;
        }}
        .source-body.open {{
            display: block;
        }}
        .source-text {{
            font-size: 12.5px;
            line-height: 1.55;
            color: #555;
            white-space: pre-wrap;
            max-height: 250px;
            overflow-y: auto;
            padding: 10px;
            background: #fafafa;
            border-radius: 6px;
            border: 1px solid #f0f0f0;
        }}
        .source-file {{
            font-size: 11px;
            color: #aaa;
            margin-top: 6px;
        }}
    </style>
    </head>
    <body>
        <div class="answer-content">
            {safe_answer}
        </div>

        <div class="sources-section">
            <div class="sources-label">Sources</div>
            {source_cards_html}
        </div>

        <script>
            function toggleSource(num) {{
                const body = document.getElementById('body-' + num);
                const chevron = document.getElementById('chevron-' + num);
                const isOpen = body.classList.contains('open');
                if (isOpen) {{
                    body.classList.remove('open');
                    chevron.classList.remove('open');
                }} else {{
                    body.classList.add('open');
                    chevron.classList.add('open');
                }}
            }}

            function showSource(num) {{
                const card = document.getElementById('source-' + num);
                const body = document.getElementById('body-' + num);
                const chevron = document.getElementById('chevron-' + num);
                if (!card) return;

                document.querySelectorAll('.source-card.highlighted').forEach(el => {{
                    el.classList.remove('highlighted');
                }});
                document.querySelectorAll('.cite.active').forEach(el => {{
                    el.classList.remove('active');
                }});

                event.target.classList.add('active');

                if (!body.classList.contains('open')) {{
                    body.classList.add('open');
                    chevron.classList.add('open');
                }}

                card.classList.add('highlighted');
                card.scrollIntoView({{ behavior: 'smooth', block: 'center' }});

                setTimeout(() => {{
                    card.classList.remove('highlighted');
                }}, 2500);
            }}
        </script>
    </body>
    </html>
    """
    return full_html
