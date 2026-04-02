import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd


@dataclass
class ScoringCoefficients:
    """Coefficients for standardized score calculation"""
    year: int
    common_coefft: float
    selection_coefft: float
    constant: float


@dataclass
class TestResult:
    """Individual test result data"""
    model_name: str
    year: int
    raw_score: int
    std_score: int
    percentile: int
    grade: int


@dataclass
class ModelStatistics:
    """Aggregated statistics for a model"""
    model_name: str
    avg_std_score: float
    avg_grade: float
    results_by_year: Dict[int, TestResult]


class ScoreCalculator:
    """Handles score calculations for different year ranges"""

    def __init__(self, data_dir: str = "data/scoring_grading_tables"):
        self.data_dir = Path(data_dir)
        self.coefficients = self._load_coefficients()

    def _load_coefficients(self) -> Dict[int, ScoringCoefficients]:
        """Load scoring coefficients for years 2022-2024"""
        return {
            2022: ScoringCoefficients(2022, 1.156, 0.913, 37.4),
            2023: ScoringCoefficients(2023, 0.980, 0.866, 35.1),
            2024: ScoringCoefficients(2024, 1.124, 0.923, 38.7)
        }

    def calculate_result(self, row: pd.Series, model_name: str) -> TestResult:
        """Calculate test result based on year"""
        year = int(row['qid_for_merge'])

        if year > 2021:
            return self._calculate_modern_score(row, model_name, year)
        else:
            return self._calculate_legacy_score(row, model_name, year)

    def _calculate_modern_score(self, row: pd.Series, model_name: str, year: int) -> TestResult:
        """Calculate score for years 2022-2024 using coefficients"""
        grading_df = self._load_grading_table(year)
        coeffs = self.coefficients[year]

        common_score = row['common_problem_score']
        choice_score = row['choice_problem_score']

        raw_score = int(row['overall_sum'])
        std_score = round(
            coeffs.common_coefft * common_score +
            coeffs.selection_coefft * choice_score +
            coeffs.constant
        )

        grading_row = grading_df[grading_df['표준점수'] == std_score]
        percentile = int(grading_row['백분위'].iloc[0])
        grade = int(grading_row['등급'].iloc[0])

        return TestResult(model_name, year, raw_score, std_score, percentile, grade)

    def _calculate_legacy_score(self, row: pd.Series, model_name: str, year: int) -> TestResult:
        """Calculate score for years 2021 and earlier using lookup table"""
        grading_df = self._load_grading_table(year)
        overall_sum = row['overall_sum']

        score_row = grading_df[grading_df['원점수'] == overall_sum]

        return TestResult(
            model_name=model_name,
            year=year,
            raw_score=int(score_row['원점수'].iloc[0]),
            std_score=int(score_row['표준점수'].iloc[0]),
            percentile=int(score_row['백분위'].iloc[0]),
            grade=int(score_row['등급'].iloc[0])
        )

    def _load_grading_table(self, year: int) -> pd.DataFrame:
        """Load grading table for a specific year"""
        file_path = self.data_dir / f"{year}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Grading table for year {year} not found at {file_path}")
        return pd.read_csv(file_path)


class LeaderboardGenerator:
    """Generates leaderboard table from scoring results"""

    def __init__(self, scoring_dir: str = "scoring_result", data_dir: str = "data/scoring_grading_tables"):
        self.scoring_dir = Path(scoring_dir)
        self.calculator = ScoreCalculator(data_dir)
        self.year_range = list(range(2024, 2014, -1))  # 2024 to 2015

    def generate_table(self) -> pd.DataFrame:
        """Generate complete leaderboard table"""
        model_stats = self._process_all_models()
        ranked_models = self._rank_models(model_stats)
        return self._build_table(ranked_models)

    def _process_all_models(self) -> Dict[str, ModelStatistics]:
        """Process all model scoring files"""
        model_stats = {}

        for file_path in self.scoring_dir.glob("*.csv"):
            model_name = file_path.stem
            try:
                model_stats[model_name] = self._process_model(file_path, model_name)
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
                continue

        return model_stats

    def _process_model(self, file_path: Path, model_name: str) -> ModelStatistics:
        """Process a single model's scoring results"""
        scoring_df = pd.read_csv(file_path)
        results = []

        for _, row in scoring_df.iterrows():
            result = self.calculator.calculate_result(row, model_name)
            results.append(result)

        # Calculate aggregated statistics
        avg_std_score = sum(r.std_score for r in results) / len(results)
        avg_grade = sum(r.grade for r in results) / len(results)
        results_by_year = {r.year: r for r in results}

        return ModelStatistics(model_name, avg_std_score, avg_grade, results_by_year)

    def _rank_models(self, model_stats: Dict[str, ModelStatistics]) -> Dict[str, ModelStatistics]:
        """Rank models by average standardized score"""
        return dict(sorted(
            model_stats.items(),
            key=lambda item: item[1].avg_std_score,
            reverse=True
        ))

    def _build_table(self, ranked_models: Dict[str, ModelStatistics]) -> pd.DataFrame:
        """Build the final leaderboard table"""
        columns = self._get_table_columns()
        table_data = []

        for rank, (model_name, stats) in enumerate(ranked_models.items(), 1):
            row_data = self._build_table_row(rank, stats)
            table_data.append(row_data)

        table = pd.DataFrame(table_data, columns=columns)
        print(table.to_markdown(index=False))
        return table

    def _get_table_columns(self) -> List[str]:
        """Get table column names"""
        return [
            'Leaderboard Rank', 'Model Name', 'Submitter Name',
            'Avg. std Score', 'Avg. Grade'
        ] + [f"{year} SAT" for year in self.year_range] + ['URL']

    def _build_table_row(self, rank: int, stats: ModelStatistics) -> Dict[str, Optional[str]]:
        """Build a single table row"""
        row = {
            'Leaderboard Rank': rank,
            'Model Name': stats.model_name,
            'Submitter Name': None,
            'Avg. std Score': round(stats.avg_std_score, 2),
            'Avg. Grade': round(stats.avg_grade, 2),
            'URL': None
        }

        # Add year-specific scores
        for year in self.year_range:
            if year in stats.results_by_year:
                result = stats.results_by_year[year]
                row[f"{year} SAT"] = f"{result.raw_score} ({result.grade})"
            else:
                row[f"{year} SAT"] = "N/A"

        return row


def main():
    """Main entry point"""
    try:
        generator = LeaderboardGenerator()
        table = generator.generate_table()
        # Uncomment to save:
        # table.to_csv("completion_table_ver2.csv", index=False)
        return table
    except Exception as e:
        print(f"Error generating leaderboard: {e}")
        raise


if __name__ == '__main__':
    table_completion = main()
