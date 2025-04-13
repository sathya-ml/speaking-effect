def parse_video_metadata(video_name: str) -> dict:
    variables = video_name.split("-")
    assert len(variables) == 7
    num_vars = list(map(int, variables))

    return {
        "modality": num_vars[0],
        "emotion": num_vars[2],
        "intensity": num_vars[3],
        "actor_num": num_vars[-1]
    }
