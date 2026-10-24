# Method lifecycle example

[Connect your agent](../../docs/DEVELOPMENT.md#connect-your-agent-loop) ·
[Run task evaluations](../../docs/QUICKSTART.md#run-a-memory-comparison)

Exercise the reference method's context reads, failure updates, and stopping
behavior before connecting your actor. The example uses fixed trial traces and
a local OpenAI-compatible response fixture; it needs no API key or benchmark
installation. Task outcomes and usage values are fixtures.

From the repository root:

```bash
python -m pip install -e '.[runtime]'
python examples/agent_loop/run.py
python examples/agent_loop/run.py --method vex2
python examples/agent_loop/run.py --method retry
```

The script reads the first trial's empty context, reports a failed attempt,
and retrieves the updated context for the next trial. After reporting success,
it stops. Use `--show-requests` to inspect the method's model requests.

For Reflexion, `benchmark="webshop"` explicitly selects the existing WebShop
writer prompt. VEX² receives a stable task description and public environment
rules, then supplies its raw joint response as system context.

For your agent, replace `FixtureClient` with your model client and the fixed
traces with executed actions and returned observations. Place
`ActorContext.system` and `.user` as shown in the
[integration guide](../../docs/DEVELOPMENT.md#place-the-returned-context-correctly),
and keep resets, the trial limit, and success stopping in your runner.
