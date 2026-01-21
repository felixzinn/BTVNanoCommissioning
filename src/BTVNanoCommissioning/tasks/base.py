import luigi


class baseparameters(luigi.Config):
    campaign = luigi.Parameter(
        description="Campaign name",
    )
    year = luigi.Parameter(
        description="Year",
    )
    version = luigi.Parameter(
        description="Output version",
    )
