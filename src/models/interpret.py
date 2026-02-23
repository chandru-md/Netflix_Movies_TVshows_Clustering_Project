### CLUSTER INTERPRETATION

def get_cluster_summary(data, top_n=5):
    summary = {}
    for cluster_id in sorted(data["cluster"].unique()):
        summary[cluster_id] = (
            data[data["cluster"] == cluster_id]["listed_in"]
            .value_counts()
            .head(top_n)
        )
    return summary
        