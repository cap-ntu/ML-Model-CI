def postprocess(results, filenames, batch_size):
    """
    Post-process results to show classifications.
    """
    if len(results) != 1:
        raise Exception("expected 1 result, got {}".format(len(results)))

    batched_result = list(results.values())[0]
    if len(batched_result) != batch_size:
        raise Exception("expected {} results, got {}".format(batch_size, len(batched_result)))
    if len(filenames) != batch_size:
        raise Exception("expected {} filenames, got {}".format(batch_size, len(filenames)))

    for (index, result) in enumerate(batched_result):
        print("Image '{}':".format(filenames[index]))
        for cls in result:
            print("    {} ({}) = {}".format(cls[0], cls[2], cls[1]))
