from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import logging
import gc
import psutil
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle

class OptimizedPopulationFeatureSelection:
    def __init__(self, data_path, output_dir, population_size=1000, initial_features=None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.population_size = population_size
        self.data = None
        self.pIC50_values = None
        self.all_features = None
        self.population = []
        self.correlation_threshold = 0.8
        self.generation = 0
        self.batch_size = max(1000, population_size // 10)
        self.save_frequency = 1
        self.memory_cleanup_frequency = 1000
        self.start_memory = 0
        self.peak_memory = 0
        self.setup_enhanced_logging()
        self.load_data()
        self.initialize_population_optimized(initial_features)

    def setup_enhanced_logging(self):
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.output_dir, f"optimized_selection_log_{timestamp}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Optimized Population-Based Selection Initialized (Population: {self.population_size})")
        self.log_system_info()

    def log_system_info(self):
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        self.logger.info("SYSTEM RESOURCES:")
        self.logger.info(f" - Total RAM: {memory.total / (1024**3):.1f} GB")
        self.logger.info(f" - Available RAM: {memory.available / (1024**3):.1f} GB")
        self.logger.info(f" - CPU Cores: {cpu_count}")
        self.logger.info(f" - Population Size: {self.population_size}")
        self.start_memory = memory.used

    def monitor_resources(self):
        memory = psutil.virtual_memory()
        current_memory = memory.used
        memory_increase = (current_memory - self.start_memory) / (1024**3)
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        self.logger.info(f"RESOURCE MONITOR:")
        self.logger.info(f" - Current RAM usage: {memory.percent:.1f}%")
        self.logger.info(f" - Memory increase: +{memory_increase:.2f} GB")
        self.logger.info(f" - Available RAM: {memory.available / (1024**3):.1f} GB")

    def load_data(self):
        try:
            self.logger.info(f"Loading data from: {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            self.pIC50_values = self.data['pIC50'].values.astype(np.float32)
            self.all_features = self.data.columns[2:-1].tolist()
            self.logger.info(f"Data loaded successfully:")
            self.logger.info(f" - Samples: {len(self.data)}")
            self.logger.info(f" - Available features: {len(self.all_features)}")
            self.logger.info(f" - Expected models per generation: {self.population_size * len(self.all_features):,}")
            gc.collect()
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def evaluate_feature_combination_optimized(self, features):
        try:
            if not features or len(features) == 0:
                return None
            feature_data = self.data[features].values.astype(np.float32)
            model = LinearRegression()
            model.fit(feature_data, self.pIC50_values)
            predictions = model.predict(feature_data)
            try:
                correlation, _ = pearsonr(self.pIC50_values, predictions)
                if np.isnan(correlation) or np.isinf(correlation):
                    correlation = -1.0
            except:
                correlation = -1.0
            mae = mean_absolute_error(self.pIC50_values, predictions)
            return {
                'features': features,
                'num_features': len(features),
                'correlation': float(correlation),
                'mae': float(mae),
                'feature_set': set(features)
            }
        except Exception as e:
            return None

    def initialize_population_optimized(self, initial_features=None):
        self.logger.info(f"Initializing population of size {self.population_size}...")
        if initial_features and len(initial_features) > 0:
            self.logger.info(f"Starting with provided features: {initial_features}")
            result = self.evaluate_feature_combination_optimized(initial_features)
            if result:
                self.population = [result]
                self.logger.info(f"Initial combination - Features: {len(initial_features)}, Correlation: {result['correlation']:.4f}")
            else:
                self.population = []
        else:
            self.population = []
        if len(self.population) < self.population_size:
            self.logger.info("Evaluating individual features to fill population...")
            individual_results = []
            for i in range(0, len(self.all_features), self.batch_size):
                batch_features = self.all_features[i:i + self.batch_size]
                self.logger.info(f" Processing batch {i//self.batch_size + 1}: features {i+1} to {min(i+self.batch_size, len(self.all_features))}")
                for feature in batch_features:
                    result = self.evaluate_feature_combination_optimized([feature])
                    if result:
                        individual_results.append(result)
                if i % (self.batch_size * 5) == 0:
                    gc.collect()
            individual_results.sort(key=lambda x: x['correlation'], reverse=True)
            remaining_slots = self.population_size - len(self.population)
            additional_population = individual_results[:remaining_slots]
            self.population.extend(additional_population)
            self.logger.info(f"Population initialized with {len(self.population)} combinations")
            if self.population:
                self.logger.info(f"Best individual: {self.population[0]['features'][0]} (r={self.population[0]['correlation']:.4f})")
            self.monitor_resources()

    def generate_candidates_optimized(self):
        candidates = []
        used_combinations = set()
        self.logger.info(f"Generating candidates from {len(self.population)} population members...")
        batch_size = max(50, self.population_size // 20)
        for batch_start in range(0, len(self.population), batch_size):
            batch_end = min(batch_start + batch_size, len(self.population))
            batch_population = self.population[batch_start:batch_end]
            self.logger.info(f" Processing population batch {batch_start//batch_size + 1}: members {batch_start+1} to {batch_end}")
            for member in batch_population:
                current_feature_set = member['feature_set']
                available_features = [f for f in self.all_features if f not in current_feature_set]
                for feature in available_features:
                    new_combination = sorted(member['features'] + [feature])
                    combination_key = tuple(new_combination)
                    if combination_key not in used_combinations:
                        used_combinations.add(combination_key)
                        candidates.append(new_combination)
            gc.collect()
        total_candidates = len(candidates)
        self.logger.info(f"Generated {total_candidates:,} unique candidate combinations")
        self.monitor_resources()
        return candidates

    def evaluate_generation_optimized(self, candidates):
        self.logger.info(f"Evaluating {len(candidates):,} candidate combinations...")
        results = []
        start_time = time.time()
        for batch_start in range(0, len(candidates), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(candidates))
            batch_candidates = candidates[batch_start:batch_end]
            batch_num = batch_start // self.batch_size + 1
            total_batches = (len(candidates) + self.batch_size - 1) // self.batch_size
            self.logger.info(f" Processing batch {batch_num}/{total_batches}: combinations {batch_start+1} to {batch_end}")
            batch_results = []
            for combination in batch_candidates:
                result = self.evaluate_feature_combination_optimized(combination)
                if result:
                    batch_results.append(result)
            results.extend(batch_results)
            elapsed = time.time() - start_time
            processed = batch_end
            rate = processed / elapsed
            remaining = len(candidates) - processed
            eta = remaining / rate if rate > 0 else 0
            self.logger.info(f" Progress: {processed:,}/{len(candidates):,} ({processed/len(candidates)*100:.1f}%) - Rate: {rate:.1f}/s - ETA: {eta:.1f}s")
            if batch_num % 5 == 0:
                gc.collect()
            self.monitor_resources()
        elapsed_time = time.time() - start_time
        self.logger.info(f"Evaluation completed in {elapsed_time:.1f} seconds")
        self.logger.info(f"Valid results: {len(results):,}/{len(candidates):,} ({len(results)/len(candidates)*100:.1f}%)")
        if not results:
            self.logger.error("No valid results from candidate evaluation!")
            return []
        self.logger.info("Sorting results by correlation...")
        results.sort(key=lambda x: x['correlation'], reverse=True)
        top_results = results[:self.population_size]
        self.logger.info(f"Selected top {len(top_results)} combinations for next generation")
        self.logger.info(f"Best: {len(top_results[0]['features'])} features, r={top_results[0]['correlation']:.4f}")
        self.logger.info(f"Worst in top {self.population_size}: {len(top_results[-1]['features'])} features, r={top_results[-1]['correlation']:.4f}")
        del results
        gc.collect()
        return top_results

    def check_convergence(self, new_population):
        if not self.population or not new_population:
            return True
        old_best = max(member['correlation'] for member in self.population)
        new_best = max(member['correlation'] for member in new_population)
        improvement = new_best - old_best
        improvement_ratio = (improvement / abs(old_best)) * 100 if old_best != 0 else 0
        self.logger.info(f"CONVERGENCE CHECK:")
        self.logger.info(f" - Previous best correlation: {old_best:.4f}")
        self.logger.info(f" - Current best correlation: {new_best:.4f}")
        self.logger.info(f" - Improvement: {improvement:.4f} ({improvement_ratio:.2f}%)")
        self.logger.info(f" - Threshold: {self.correlation_threshold}%")
        if improvement_ratio < self.correlation_threshold:
            self.logger.info(f"Convergence criteria met - stopping evolution")
            return False
        return True

    def save_generation_results_enhanced(self, generation, population):
        if not population:
            return
        summary_file = os.path.join(self.output_dir, "generation_summary.csv")
        correlations = [member['correlation'] for member in population]
        maes = [member['mae'] for member in population]
        feature_counts = [member['num_features'] for member in population]
        summary_data = {
            'generation': generation,
            'population_size': len(population),
            'best_correlation': max(correlations),
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'worst_correlation': min(correlations),
            'best_mae': min(maes),
            'mean_mae': np.mean(maes),
            'worst_mae': max(maes),
            'min_features': min(feature_counts),
            'mean_features': np.mean(feature_counts),
            'max_features': max(feature_counts),
            'timestamp': datetime.now().isoformat(),
            'memory_gb': psutil.virtual_memory().used / (1024**3)
        }
        summary_df = pd.DataFrame([summary_data])
        if os.path.exists(summary_file):
            summary_df.to_csv(summary_file, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(summary_file, index=False)
        if generation % self.save_frequency == 0 or population[0].get('is_final', False):
            top_population = population[:50]
            detailed_file = os.path.join(self.output_dir, f"top50_population_gen_{generation}.csv")
            detailed_data = []
            for idx, member in enumerate(top_population):
                row = {
                    'rank': idx + 1,
                    'num_features': member['num_features'],
                    'correlation': member['correlation'],
                    'mae': member['mae'],
                    'features': '|'.join(member['features'])
                }
                detailed_data.append(row)
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_csv(detailed_file, index=False)
            self.logger.info(f"Top 50 results saved: {detailed_file}")

    def run_optimized_evolution(self, max_generations=15):
        start_time = time.time()
        self.logger.info("="*80)
        self.logger.info("STARTING OPTIMIZED POPULATION-BASED FEATURE SELECTION")
        self.logger.info("="*80)
        self.logger.info(f"Population size: {self.population_size:,}")
        self.logger.info(f"Available features: {len(self.all_features):,}")
        self.logger.info(f"Expected models per generation: {self.population_size * len(self.all_features):,}")
        self.logger.info(f"Batch size: {self.batch_size:,}")
        self.logger.info(f"Max generations: {max_generations}")
        if not hasattr(self, 'population') or not self.population:
            self.initialize_population_optimized()
        if not self.population:
            self.logger.error("Failed to initialize population")
            return None
        self.save_generation_results_enhanced(0, self.population)
        generation = 0
        while generation < max_generations:
            generation += 1
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"GENERATION {generation}")
            self.logger.info(f"{'='*60}")
            candidates = self.generate_candidates_optimized()
            if not candidates:
                self.logger.info("No more candidates available")
                break
            new_population = self.evaluate_generation_optimized(candidates)
            if not new_population:
                self.logger.error("No valid combinations found")
                break
            if not self.check_convergence(new_population):
                break
            self.population = new_population
            self.save_generation_results_enhanced(generation, self.population)
            elapsed = time.time() - start_time
            avg_time = elapsed / generation
            self.logger.info(f"Time: {elapsed:.1f}s elapsed, {avg_time:.1f}s per generation")
            self.monitor_resources()
        total_time = time.time() - start_time
        self.save_final_results_enhanced(self.population, generation, total_time)
        return self.population[0] if self.population else None

    def save_final_results_enhanced(self, final_population, total_generations, total_time):
        if not final_population:
            return
        best_result = final_population[0]
        best_result['is_final'] = True
        optimal_file = os.path.join(self.output_dir, "optimal_feature_combinations_top1000.txt")
        with open(optimal_file, 'w') as f:
            f.write("OPTIMIZED POPULATION-BASED FEATURE SELECTION RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Algorithm: Population-based forward selection (Optimized)\n")
            f.write(f"Population size: {self.population_size:,}\n")
            f.write(f"Total generations: {total_generations}\n")
            f.write(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)\n")
            f.write(f"Peak memory usage: {self.peak_memory / (1024**3):.2f} GB\n")
            f.write(f"Improvement threshold: {self.correlation_threshold}%\n\n")
            f.write("OPTIMAL FEATURE COMBINATION (Rank 1):\n")
            f.write("-" * 50 + "\n")
            f.write(f"Features: {best_result['num_features']}\n")
            f.write(f"Correlation: {best_result['correlation']:.4f}\n")
            f.write(f"MAE: {best_result['mae']:.4f}\n\n")
            f.write("Feature list:\n")
            for i, feature in enumerate(best_result['features'], 1):
                f.write(f"{i:2d}. {feature}\n")
            f.write(f"\n\nTOP 20 FEATURE COMBINATIONS:\n")
            f.write("=" * 60 + "\n")
            for rank, member in enumerate(final_population[:20], 1):
                f.write(f"\nRank {rank}:\n")
                f.write(f" Features: {member['num_features']}\n")
                f.write(f" Correlation: {member['correlation']:.4f}\n")
                f.write(f" MAE: {member['mae']:.4f}\n")
        self.logger.info(f"Final results saved to: {optimal_file}")
        self.save_generation_results_enhanced(total_generations, final_population)
        # =========== INSERTED CODE TO SAVE BEST MODEL ===========
        optimal_features = best_result['features']
        model = LinearRegression()
        feature_data = self.data[optimal_features].values.astype(np.float32)
        model.fit(feature_data, self.pIC50_values)
        with open(os.path.join(self.output_dir, "best_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        # =======================================================

def main():
    DATA_PATH = r"D:\Afra J\IITM\MD5001\Project - LATS1\LATS1_train.csv"
    OUTPUT_DIR = r"D:\Afra J\IITM\MD5001\Project - LATS1\Results_2"
    POPULATION_SIZE = 1000
    CORRELATION_THRESHOLD = 0.05
    MAX_GENERATIONS = 15
    INITIAL_FEATURES = []
    print("🚀 STARTING OPTIMIZED TOP 1000 POPULATION SELECTION")
    print("="*60)
    print(f"Population size: {POPULATION_SIZE:,}")
    print(f"Expected models per generation: ~278,000")
    print("="*60)
    try:
        selector = OptimizedPopulationFeatureSelection(
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
            population_size=POPULATION_SIZE,
            initial_features=INITIAL_FEATURES
        )
        selector.correlation_threshold = CORRELATION_THRESHOLD
        best_result = selector.run_optimized_evolution(max_generations=MAX_GENERATIONS)
        if best_result:
            print("\n" + "="*70)
            print("OPTIMIZED TOP 1000 SELECTION COMPLETED!")
            print("="*70)
            print(f"Optimal features: {best_result['num_features']}")
            print(f"Best correlation: {best_result['correlation']:.4f}")
            print(f"Best MAE: {best_result['mae']:.4f}")
            print(f"Results in: {OUTPUT_DIR}")
        else:
            print("Selection completed but no result found")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
