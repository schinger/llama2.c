## llama2 Pre-training/finetuning, PPO(RLHF), Inference

## custom tokenizers

In everything above, we've assumed the custom Lllama 2 tokenizer with 32,000 tokens. However, in many boutique LLMs, using vocabulary this big might be an overkill. If you have a small application you have in mind, you might be much better off training your own tokenizers. This can make everything nicer - with smaller vocabs your model has fewer parameters (because the token embedding table is a lot smaller), the inference is faster (because there are fewer tokens to predict), and your average sequence length per example could also get smaller (because the compression is a lot more efficient on your data). So let's see how we train a custom tokenizer.

By default, to pretokenize the tinystories dataset we had to run, in order:

```
python tinystories.py download
python tinystories.py pretokenize
```

The `pretokenize` stage here loads the Llama 2 tokenizer (vocab size 32,000) and uses it to convert the downloaded text into integers, and saves that to file. We now change this as follows, to train an example 4096-token tokenizer:

```
python tinystories.py download
python tinystories.py train_vocab --vocab_size=4096
python tinystories.py pretokenize --vocab_size=4096
```

The `train_vocab` stage will call the `sentencepiece` library to train the tokenizer, storing it in a new file `data/tok4096.model`. I tried to reproduce as well as I could the settings that (I think) Meta used to train their vocabulary. This uses the Byte Pair Encoding algorithm that starts out with raw utf8 byte sequences of the text data and then iteratively merges the most common consecutive pairs of tokens to form the vocabulary. Inspect the `tinystories.py` file - the custom tokenizers are stored in a special directory structure indexed by the vocab size.

A quick note of interest is that vocab size of 4096 trained specifically on tinystories creates integer sequences with about the same sequence length per example as the default Llama 2 tokenizer of 32000 tokens! This means that our custom, tailored tokenizer is a lot better adapted to our specific text, and can compress it very effectively. So our trained models are smaller and faster.

Now that we have pretokenized the dataset with our custom tokenizer, we can train the model. The training script `train.py` doesn't care about the exact tokens, it only cares about the vocabulary size so it can correctly initialize the model. So when training your model, make sure to pass in

```
python train.py --vocab_source=custom --vocab_size=4096
```

(The defaults are `llama2` and `32000` respectively, which indicates the default Llama 2 tokenizer). This trains the model. Finally we are ready to run inference with our `run.c` script. For that we need two things. Number one, we have to export our tokenizer in the `.bin` format, do that with:

```
python tokenizer.py --tokenizer-model=data/tok4096.model
```

This writes the tokenizer to `data/tok4096.bin`. Now we can run inference, pointing it to this tokenizer using the `-z` flag:

```
./run out/model.bin -z data/tok4096.bin
```

This should print the samples. If you leave out the `-z` flag, it will use the default Llama 2 tokenizer, which would generate a good sequence of integers, but they would get translated using a different vocabulary to text, so it would look like gibberish.


## Just learn the target length

```
python train.py     --out_dir="stories260K"     --batch_size=50     --max_seq_len=512     --gradient_accumulation_steps=1     --vocab_source="custom"     --vocab_size=512     --dim=64     --n_layers=5     --n_heads=8     --n_kv_heads=4     --multiple_of=4     --learning_rate=1e-4     --dropout=0.00     --weight_decay=0.01     --max_iters=200000     --beta2=0.99     --warmup_iters=1000     --eval_interval=20     --eval_iters=5     --compile=False    --device=cpu    --eval_only=False   --init_from="resume" --ppo=True  --decay_lr=False  --always_save_checkpoint=True  --start_len=1  --init_kl_coef=0.0
```
after run about several iterations, it can reach the length goal.
```
q+r lengths [201, 200, 200, 201, 199, 199, 200, 201, 200, 200]
q+r lengths avg 200.175
98147 | reward -0.0087 | lr 1.000000e-04 | 154092.34ms
{
    "ppo/loss/policy": -0.004611059091985226,
    "ppo/loss/value": 7.655927038285881e-05,
    "ppo/loss/total": -0.004603402689099312,
    "ppo/policy/entropy": 0.36575931310653687,
    "ppo/policy/policykl": 0.0045680576004087925,
    "ppo/policy/clipfrac": 0.022210117429494858,
    "ppo/policy/advantages_mean": 0.00017487886361777782,
    "ppo/returns/mean": -0.008375758305191994,
    "ppo/returns/var": 0.00014193591778166592,
    "ppo/val/vpred": -0.008607665076851845,
    "ppo/val/error": 0.00015311854076571763,
    "ppo/val/clipfrac": 0.0,
    "ppo/val/mean": -0.008562136441469193,
    "ppo/val/var": 3.553760689101182e-05,
    "ppo/learning_rate": 0.0001,
    "ppo/kl_ref": 609.431640625,
    "ppo/reward_all": -0.008749999105930328,
    "time/ppo/forward_pass": 22.32808017730713,
    "time/ppo/compute_rewards": 0.04999852180480957,
    "time/ppo/compute_advantages": 0.029018878936767578,
    "time/ppo/optimize_step": 81.42295384407043,
    "time/ppo/calc_stats": 0.004998207092285156,
    "time/ppo/total": 103.83504962921143
}
```

## Nearly reach the target length and generate fluent sentences

```
python train.py     --out_dir="stories260K"     --batch_size=50     --max_seq_len=512     --gradient_accumulation_steps=1     --vocab_source="custom"     --vocab_size=512     --dim=64     --n_layers=5     --n_heads=8     --n_kv_heads=4     --multiple_of=4     --learning_rate=1e-4     --dropout=0.00     --weight_decay=0.01     --max_iters=200000     --beta2=0.99     --warmup_iters=1000     --eval_interval=20     --eval_iters=5     --compile=False    --device=cpu    --eval_only=False   --init_from="resume" --ppo=True  --decay_lr=False  --always_save_checkpoint=True  --start_len=30
```

initial statistics:
```
q+r lengths [252, 297, 328, 328, 328, 328, 328, 328, 328, 328]
q+r lengths avg 304.1
98000 | reward -1.0687 | lr 1.000000e-04 | 280442.50ms
{
    "ppo/loss/policy": -0.014800487086176872,
    "ppo/loss/value": 0.060744259506464005,
    "ppo/loss/total": -0.008726062253117561,
    "ppo/policy/entropy": 1.3361481428146362,
    "ppo/policy/policykl": 0.004431259818375111,
    "ppo/policy/clipfrac": 0.04800604283809662,
    "ppo/policy/advantages_mean": 0.00011433474719524384,
    "ppo/returns/mean": -0.08783192932605743,
    "ppo/returns/var": 0.04501805081963539,
    "ppo/val/vpred": -0.04483073577284813,
    "ppo/val/error": 0.12132434546947479,
    "ppo/val/clipfrac": 0.00803137756884098,
    "ppo/val/mean": -0.019085045903921127,
    "ppo/val/var": 0.10107368230819702,
    "ppo/learning_rate": 0.0001,
    "ppo/kl_ref": 0.0,
    "ppo/reward_all": -1.0686500072479248,
    "time/ppo/forward_pass": 28.0527503490448,
    "time/ppo/compute_rewards": 0.046872615814208984,
    "time/ppo/compute_advantages": 0.031641483306884766,
    "time/ppo/optimize_step": 103.44326996803284,
    "time/ppo/calc_stats": 0.015612363815307617,
    "time/ppo/total": 131.59014678001404
}
```
after run about 49 iterations (98049), it can reach the avg 200.235 length:
```
q+r lengths [169, 230, 199, 135, 197, 172, 297, 189, 162, 201]
q+r lengths avg 200.235
98049 | reward -0.3672 | lr 1.000000e-04 | 212372.84ms
{
    "ppo/loss/policy": -0.015469990670681,
    "ppo/loss/value": 0.03054601326584816,
    "ppo/loss/total": -0.012415390461683273,
    "ppo/policy/entropy": 1.2525558471679688,
    "ppo/policy/policykl": 0.004105564206838608,
    "ppo/policy/clipfrac": 0.045593008399009705,
    "ppo/policy/advantages_mean": 0.0001622058916836977,
    "ppo/returns/mean": -1.400254726409912,
    "ppo/returns/var": 0.0390283428132534,
    "ppo/val/vpred": -1.3735605478286743,
    "ppo/val/error": 0.06109202653169632,
    "ppo/val/clipfrac": 0.0,
    "ppo/val/mean": -1.363938331604004,
    "ppo/val/var": 0.04070740565657616,
    "ppo/learning_rate": 0.0001,
    "ppo/kl_ref": 7.249481201171875,
    "ppo/reward_all": -1.81709623336792,
    "time/ppo/forward_pass": 26.893563508987427,
    "time/ppo/compute_rewards": 0.04730224609375,
    "time/ppo/compute_advantages": 0.031235456466674805,
    "time/ppo/optimize_step": 97.83660507202148,
    "time/ppo/calc_stats": 0.0,
    "time/ppo/total": 124.80870628356934
}
```
set temperature to 0.4, it can generate fluent sentences (length: 218):
```
Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. One day, Lily's mom asked her to clean up her toys and they went to the park to play. Lily was very happy and wanted to play with her toys.
As they were walking, Lily saw a big, red balloon in the park. She asked her mom, "What's wrong, mom?" Her mom said, "I'm sorry, Lily. I didn't know what to do."
Lily said, "I don't know, I will help you." Her mom smiled and said, "Yes, Lily. You are a good friend."
```
## Generate
```
python sample.py --checkpoint=stories260K/ppo_ckpt.pt --tokenizer=stories260K/tok512.model --temperature=0.0 --max_new_tokens=500
```
## License

MIT
