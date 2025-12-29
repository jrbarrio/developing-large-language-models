for batch in tqdm(ppo_trainer.dataloader): 

    # Generate responses for the given queries using the trainer
    response_tensors = ppo_trainer.generate(batch["input_ids"])

    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    texts = [q + r for q, r in zip(batch["query"], batch["response"])]

    rewards = reward_model(texts)

    # Training PPO step with the query, responses ids, and rewards
    stats = ppo_trainer.step(batch["input_ids"], response_tensors, rewards)

    ppo_trainer.log_stats(stats, batch, rewards)