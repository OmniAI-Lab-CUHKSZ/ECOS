"""This run permits increased free memory during vLLM profiling, as requested."""
import ast
import inspect
import textwrap

def rewrite_source(source):
    tree=ast.parse(textwrap.dedent(source))
    for function in tree.body:
        if not isinstance(function,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
        for index,node in enumerate(function.body):
            if isinstance(node,ast.Assert) and 'Error in memory profiling.' in (ast.get_source_segment(textwrap.dedent(source),node) or ''):
                replacement=ast.parse("if not (True):\n    logger.warning('Free GPU memory increased during profiling; continuing as requested')").body[0]
                replacement.test=ast.UnaryOp(op=ast.Not(),operand=node.test)
                function.body[index]=ast.copy_location(replacement,node)
    return ast.fix_missing_locations(tree)


def rewrite_sleep_source(source):
    source=textwrap.dedent(source)
    tree=ast.parse(source);changed=0
    for function in tree.body:
        if not isinstance(function,ast.FunctionDef):continue
        for index,node in enumerate(function.body):
            if isinstance(node,ast.Assert) and 'Memory usage increased after sleeping.' in (ast.get_source_segment(source,node) or ''):
                replacement=ast.parse("if not (True):\n    logger.warning('Free GPU memory decreased during sleep; offload completed; continuing as requested')").body[0]
                replacement.test=ast.UnaryOp(op=ast.Not(),operand=node.test)
                function.body[index]=ast.copy_location(replacement,node);changed+=1
    assert changed==1, 'Expected the installed sleep memory-accounting assertion'
    return ast.fix_missing_locations(tree)


def install():
    from vllm.v1.worker.gpu_worker import Worker
    if not getattr(Worker,'_alfworld_allows_profile_free_increase',False):
        original=inspect.unwrap(Worker.determine_available_memory)
        namespace=dict(original.__globals__)
        tree=rewrite_source(inspect.getsource(original))
        exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace)
        Worker.determine_available_memory=namespace['determine_available_memory']
        Worker._alfworld_allows_profile_free_increase=True
        print('VLLM_PROFILE_FREE_INCREASE_ALLOWED_REAL_OOM_UNCHANGED',flush=True)
    if not getattr(Worker,'_alfworld_allows_sleep_free_decrease',False):
        original=inspect.unwrap(Worker.sleep)
        namespace=dict(original.__globals__)
        tree=rewrite_sleep_source(inspect.getsource(original))
        exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace)
        Worker.sleep=namespace['sleep']
        Worker._alfworld_allows_sleep_free_decrease=True
        print('VLLM_SLEEP_ACCOUNTING_INCREASE_ALLOWED_REAL_OOM_UNCHANGED',flush=True)
