"""
diagrams.py

The "multi-modal" part of Phase 1: schematics and wiring diagrams don't
have parseable table structure like the parameter/fault tables do, so
instead of parsing them as data, we extract them as images + surrounding
text context, and optionally ask a vision-capable model to describe them.

Two-step design, kept separate on purpose:
  extract_diagrams() - pure, offline, no API calls. Pulls every embedded
                        image out of a page range and saves it alongside
                        the page's plain text (used as caption context,
                        since these manuals describe a figure in the
                        surrounding paragraph rather than in a dedicated
                        caption field).
  describe_diagram()  - optional, calls the Claude API (vision) to turn
                        an extracted image + its context into a short
                        text description that CAN be embedded and
                        retrieved like any other chunk. Requires
                        ANTHROPIC_API_KEY; skipped automatically if unset,
                        so extraction always works even with no API key.

Usage (inside Docker):
    docker compose exec dev python src/ingest/diagrams.py data/raw/<file>.pdf 1 48 --out data/processed/diagrams
    docker compose exec dev python src/ingest/diagrams.py data/raw/<file>.pdf 1 48 --describe
"""

import argparse
import base64
import json
import os
import sys

import fitz  # PyMuPDF


def extract_diagrams(pdf_path, first_page, last_page, out_dir):
    """
    For each page in range, extract embedded images and the page's full
    text as context. Returns a list of dicts:

        {
          "id": "p12_img0",
          "page": 12,
          "image_path": "data/processed/diagrams/p12_img0.png",
          "context_text": "...",
        }

    Skips pages with no embedded images (most pages in a text-heavy
    manual). A page with N images produces N records, all sharing the
    same context_text, since manuals rarely caption images individually.
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    records = []

    for pno in range(first_page, last_page + 1):
        page = doc[pno - 1]
        images = page.get_images(full=True)
        if not images:
            continue

        context_text = page.get_text().strip()

        for idx, img in enumerate(images):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:
                    # CMYK (or worse) can't be saved directly as PNG
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                image_id = f"p{pno}_img{idx}"
                image_path = os.path.join(out_dir, f"{image_id}.png")
                pix.save(image_path)
                pix = None

                records.append({
                    "id": image_id,
                    "page": pno,
                    "image_path": image_path,
                    "context_text": context_text,
                })
            except Exception as e:
                print(f"Failed to extract image on page {pno} (xref {xref}): {e}",
                      file=sys.stderr)

    return records


def describe_diagram(record, model="claude-sonnet-5"):
    """
    Ask a vision-capable Claude model to describe what a diagram shows,
    using the surrounding page text as context. Adds a 'description'
    field to the record and returns it.

    Requires ANTHROPIC_API_KEY (see .env). If it isn't set, this is a
    no-op that returns the record unchanged - extraction still works
    fully offline, this step is opt-in.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"Skipping description for {record['id']}: ANTHROPIC_API_KEY not set",
              file=sys.stderr)
        return record

    import anthropic
    client = anthropic.Anthropic()

    with open(record["image_path"], "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "This is a diagram or schematic from an industrial equipment "
        "manual. The surrounding page text is:\n\n"
        f"{record['context_text'][:1500]}\n\n"
        "Describe what this diagram shows in 2-3 sentences, focused on "
        "information useful for equipment diagnosis: what components it "
        "labels, what connections or signal paths it shows, and any "
        "codes or terminal numbers visible."
    )

    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    record["description"] = response.content[0].text
    return record


def diagram_to_chunk(record, manual="unknown-manual"):
    """
    Turn one described diagram into an embeddable chunk, in the same
    shape as parent_to_chunk()/event_to_chunk() so all three chunk
    types can be mixed into one vector store. Only meaningful after
    describe_diagram() has added a 'description' field.
    """
    text = (
        f"{manual} > diagram > page {record['page']}\n"
        f"Diagram: {record.get('description', '(no description generated)')}"
    )
    return {
        "id": f"{manual}-diagram-{record['id']}",
        "text": text,
        "metadata": {
            "manual": manual,
            "section": "diagram",
            "page": record["page"],
            "image_path": record["image_path"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract diagrams/images from a PDF manual.")
    parser.add_argument("pdf_path")
    parser.add_argument("first_page", type=int)
    parser.add_argument("last_page", type=int)
    parser.add_argument("--out", default="data/processed/diagrams")
    parser.add_argument("--manual", default="unknown-manual")
    parser.add_argument("--describe", action="store_true",
                         help="also call the vision model to describe each image "
                              "(requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    recs = extract_diagrams(args.pdf_path, args.first_page, args.last_page, args.out)
    print(f"{len(recs)} images extracted")

    if recs:
        print("\nExample record (context truncated):")
        preview = {**recs[0], "context_text": recs[0]["context_text"][:200]}
        print(json.dumps(preview, indent=2))

        if args.describe:
            described = describe_diagram(recs[0], )
            print("\nExample description:")
            print(described.get("description", "(none - check ANTHROPIC_API_KEY)"))

            print("\nExample chunk:")
            print(json.dumps(diagram_to_chunk(described, manual=args.manual), indent=2))
