from azure.ai.ml import dsl, Input

from privacy_estimates.experiments.loaders import SingleGameLoader
from privacy_estimates.experiments.components import aggregate_output


class RunManyGamesLoader:
    def __init__(self, single_game: SingleGameLoader, num_games: int):
        self.single_game = single_game
        self.num_games = num_games

    def load(self, train_data: Input, validation_data: Input, base_seed: int):
        @dsl.pipeline(name=f"run_{self.num_games}_games")
        def pipeline(train_data: Input, validation_data: Input, base_seed: int):
            scores = []
            challenge_bits = []
            dp_parameters = []
            for i in range(0, self.num_games):
                seed = base_seed + i
                game = self.single_game.load(seed=seed, train_data=train_data, validation_data=validation_data)

                scores.append(game.outputs.scores)
                challenge_bits.append(game.outputs.challenge_bits)
                if hasattr(game.outputs, "dp_parameters"):
                    dp_parameters.append(game.outputs.dp_parameters) 

            results = {
                "scores": aggregate_output(scores, aggregator="concatenate_datasets"),
                "challenge_bits": aggregate_output(challenge_bits, aggregator="concatenate_datasets"),
            }
            if len(dp_parameters) > 0:
                results["dp_parameters"] = aggregate_output(dp_parameters, aggregator="assert_json_equal")
            return results
        return pipeline(train_data=train_data, validation_data=validation_data, base_seed=base_seed)
