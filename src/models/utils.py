def to_dialog_template(queries):
    dialog_template = []
    for q in queries:
        dialog_template.append(
            [
                {
                    "role": "user",
                    "content": q,
                }
            ]
        )
    return dialog_template

def generate_closed_instr(instr):
    return "[INST] " + instr + " [/INST]"
    # return "<s>[INST] " + instr + " [/INST]"
