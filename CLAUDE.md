
# vLLM Extended Agent
## Overview
This agent is specialized for working with an extended version of vLLM that enables dynamic pipeline parallelism configuration switching and KV cache migration capabilities. The agent has deep knowledge of the codebase architecture, experimental setup, and best practices for development and testing.

This repository is an **extended version of vLLM** that addresses limitations in the original implementation:
- **Problem**: Traditional vLLM cannot perform dynamic pipeline parallelism configuration switching during runtime
- **Solution**: This repository extends vLLM to support dynamic reconfiguration of pipeline parallelism stages, enabling flexible model serving and efficient resource utilization
- **Additional Features**: Includes experimental environment code for testing and validation

## Basic Rules
1. In the process of solving a problem, you will find youself generating all sorts of tools to analyse log, trying to use the exisiting tools provided in the **./vllm_exp/tools**, if you need a new tools, please think of maybe adding feature to exising one. Try to make it easy to maintain, you should think these set of tools as a framework of log analysing. If the things you want to do can only be done by temporary scripts and not be easily integrated to existing tools, you should place it in /tmp directory.
2. Keeps the repository clean and prevents accidental commits of temporary code.
3. When investigating whether a log file has error, rather than print the tail of the log, you should search whether there is errors happening in the log.
4. When answering me or when generating a report, use *Chinese*
5. 在当前的spartan环境下，文件系统都是共享的，所以你修改了一个地方的文件，不需要在所有节点上修改。
6. Whenever you think the seesion you are running have valuable informations, keep it as a .md file in the ./memory directory. Don't overdo it.
7. When drawing, output should be put into the sosp paper's draw directory.

## Delegating Subagents
Whenever you think is appropriate, you can delegate tasks to subagents to save your context window. Here are some example scenarios:
   Research before implementation
   Parallel code analysis
   Explore multiple solutions
   Code review with specialized focus


## 命令执行规则

1. **永远追踪命令**: 执行命令时必须使用 `timeout: 0`，确保完整等待命令执行完成
2. **禁止擅自后台运行**: 如果认为必须使用 `isBackground: true` 或不追踪命令，**必须先询问用户**
3. 不要使用&来后台运行任务
4. 永远不要重启ray cluster，当你觉得需要重启的时候，请你pass给我让我手动进行这个操作。
5. If the rank 0 in the config is on remote log, you should ssh to the remote server and run the commands.

## Development
### Code Modification Guidelines
1. Inheritance Over Modification: Extend existing classes rather than modifying original vLLM code
2. Naming Conventions: 
   - Prefix dynamic extensions with `dynamic_`
   - Prefix flexible KV cache code with `flexi_`
3. Documentation: Document all extensions with clear purpose and usage
4. Testing: Always test changes using the experimental environment
5. When adding new variables to vllm, making sure to also add them to the experiment framework exp_vllm to make sure user can directly configure them via the experiment configuration files. Making sure not using environment variables, add them to the data.py and pass them through vllm's args.
6. You should always write most succint, clear and elegent code. Don't over engineer stuff.
7. When writing code, you should follow the *fail quick* principle, we are not developing production application, we want the problem to be opposed early.


# Subagent Instructions(important)
You are not having a large context window(100k ~ 200k), You should delegate subagents to do its tasks whenever it is possible to save context window.