import vectorlab as vl

urls = [
    f'www.test.com/samples/{i}'
    for i in range(20)
]

deg_graph = vl.stats.DegreeGraph(
    split_token='/',
    wild_token='*',
    deg_threshold=10,
    failed_safe='invalid'
)

transformed_urls = deg_graph.fit_transform(
    urls
)

print(transformed_urls)
