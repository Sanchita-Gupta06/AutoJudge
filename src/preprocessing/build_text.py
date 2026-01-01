def build_full_text(row):
    return "\n".join([
        str(row.get("title", "")),
        str(row.get("description", "")),
        str(row.get("input_description", "")),
        str(row.get("output_description", ""))
    ])