    qa_result = await asyncio.to_thread(
        ask, question, summary, raw_chunks, None
    )