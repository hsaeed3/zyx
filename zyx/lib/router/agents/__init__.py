__all__ = [
    "judge",
    "plan",
    "query",
    "scrape",
    "solve"
]


from .._router import router


class judge(router):
    pass


judge.init("zyx.resources.completions.agents.judge", "judge")


class plan(router):
    pass


plan.init("zyx.resources.completions.agents.plan", "plan")


class query(router):
    pass


query.init("zyx.resources.completions.agents.query", "query")


class scrape(router):
    pass


scrape.init("zyx.resources.completions.agents.scrape", "scrape")


class solve(router):
    pass


solve.init("zyx.resources.completions.agents.solve", "solve")
