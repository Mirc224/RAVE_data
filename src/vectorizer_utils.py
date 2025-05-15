def create_binary_vector(ordered_attributes: list[str], sentences_attributes: set[str]):
    return [1 if attribute in sentences_attributes else 0 for attribute in ordered_attributes]