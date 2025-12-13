"""
Тестовый пайплайн для оценки агентов EvaluatorAgent и TutorAgent.

Использование:
    python tests/test_agents.py --agent tutor --data data/dataset/student_questions.csv
    python tests/test_agents.py --agent evaluator --data data/dataset/attack_prompts.csv
    python tests/test_agents.py --agent all --output results/
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from agents.tutor.tutor_agent import TutorAgent
    from agents.evaluator.evaluator_agent import EvaluatorAgent
    from config import get_tutor_config, get_evaluator_config, get_api_key
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что агенты находятся в правильных директориях")
    sys.exit(1)


class AgentTester:
    """Класс для тестирования агентов на тестовых данных."""
    
    def __init__(self, agent_type: str, output_dir: str = "results"):
        """
        Инициализация тестера.
        
        Args:
            agent_type: Тип агента ('tutor' или 'evaluator')
            output_dir: Директория для сохранения результатов
        """
        self.agent_type = agent_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Инициализация агента
        self.agent = self._init_agent()
        
        # Результаты тестирования
        self.results: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
    
    def _init_agent(self):
        """Инициализация агента."""
        api_key = get_api_key()
        if not api_key:
            raise ValueError(
                "Не установлен API ключ. Установите переменную окружения OPENAI_API_KEY или API_KEY"
            )
        
        if self.agent_type == "tutor":
            # Используем значения по умолчанию из config, передаем только api_key
            return TutorAgent(api_key=api_key)
        elif self.agent_type == "evaluator":
            # Используем значения по умолчанию из config, передаем только api_key
            return EvaluatorAgent(api_key=api_key)
        else:
            raise ValueError(f"Неизвестный тип агента: {self.agent_type}")
    
    def test_tutor_agent(self, data_file: str):
        """Тестирование TutorAgent на данных из CSV."""
        print(f"\n{'='*60}")
        print(f"Тестирование TutorAgent")
        print(f"{'='*60}")
        print(f"Загрузка данных из: {data_file}")
        
        # Загрузка данных
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            test_cases = list(reader)
        
        print(f"Загружено тестовых случаев: {len(test_cases)}")
        print(f"\nНачало тестирования...\n")
        
        # Статистика
        total = len(test_cases)
        correct_stage = 0
        correct_tools = 0
        used_guiding_questions = 0
        
        for i, case in enumerate(test_cases, 1):
            print(f"[{i}/{total}] Обработка вопроса ID: {case['question_id']}")
            
            # Подготовка входных данных
            assignment_type = case['assignment_type']
            student_question = case['student_question']
            expected_stage = case['student_stage']
            expected_tools = case.get('expected_tools', '').split(',')
            expected_guiding = case.get('expected_guiding_question', 'false').lower() == 'true'
            
            # Требования к заданию
            assignment_requirements = {
                "description": f"Задание по {assignment_type}",
                "success_criteria": {"min_completeness": 0.8}
            }
            
            try:
                # Вызов агента
                result = self.agent.help_student(
                    assignment_type=assignment_type,
                    student_question=student_question,
                    assignment_requirements=assignment_requirements,
                    student_current_solution=None
                )
                
                # Анализ результата
                predicted_stage = result.get('stage', 'unknown')
                used_tools = result.get('tools_used', [])
                used_guiding = 'ask_guiding_question' in used_tools
                
                # Проверка метрик
                stage_correct = predicted_stage == expected_stage
                tools_correct = any(tool in used_tools for tool in expected_tools if tool)
                guiding_correct = used_guiding == expected_guiding
                
                if stage_correct:
                    correct_stage += 1
                if tools_correct:
                    correct_tools += 1
                if used_guiding:
                    used_guiding_questions += 1
                
                # Сохранение результата
                self.results.append({
                    'question_id': case['question_id'],
                    'assignment_type': assignment_type,
                    'student_question': student_question,
                    'expected_stage': expected_stage,
                    'predicted_stage': predicted_stage,
                    'stage_correct': stage_correct,
                    'expected_tools': expected_tools,
                    'used_tools': used_tools,
                    'tools_correct': tools_correct,
                    'expected_guiding': expected_guiding,
                    'used_guiding': used_guiding,
                    'guiding_correct': guiding_correct,
                    'result': result
                })
                
                print(f"  ✓ Этап: {predicted_stage} (ожидалось: {expected_stage}) {'✓' if stage_correct else '✗'}")
                print(f"  ✓ Инструменты: {used_tools}")
                
            except Exception as e:
                print(f"  ✗ Ошибка: {e}")
                self.results.append({
                    'question_id': case['question_id'],
                    'error': str(e)
                })
        
        # Вычисление метрик
        self.metrics = {
            'total_cases': total,
            'stage_accuracy': correct_stage / total if total > 0 else 0,
            'tool_selection_accuracy': correct_tools / total if total > 0 else 0,
            'guiding_questions_usage': used_guiding_questions / total if total > 0 else 0,
            'correct_stage': correct_stage,
            'correct_tools': correct_tools,
            'used_guiding_questions': used_guiding_questions
        }
        
        print(f"\n{'='*60}")
        print("Результаты тестирования TutorAgent:")
        print(f"{'='*60}")
        print(f"Всего тестовых случаев: {total}")
        print(f"Точность определения этапа: {self.metrics['stage_accuracy']:.2%}")
        print(f"Точность выбора инструментов: {self.metrics['tool_selection_accuracy']:.2%}")
        print(f"Использование доспрашивания: {self.metrics['guiding_questions_usage']:.2%}")
    
    def test_evaluator_agent(self, data_file: str):
        """Тестирование EvaluatorAgent на данных из CSV."""
        print(f"\n{'='*60}")
        print(f"Тестирование EvaluatorAgent")
        print(f"{'='*60}")
        print(f"Загрузка данных из: {data_file}")
        
        # Загрузка данных
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            test_cases = list(reader)
        
        print(f"Загружено тестовых случаев: {len(test_cases)}")
        print(f"\nНачало тестирования...\n")
        
        # Статистика
        total = len(test_cases)
        correct_stage = 0
        correct_validator = 0
        score_in_range = 0
        used_llm_analysis = 0
        
        for i, case in enumerate(test_cases, 1):
            print(f"[{i}/{total}] Обработка промпта ID: {case['prompt_id']}")
            
            # Подготовка входных данных
            assignment_type = case['assignment_type']
            attack_prompt = case['attack_prompt']
            expected_status = case['status']
            score_range = case.get('score_range', '0-100')
            extraction_successful = case.get('extraction_successful', 'false').lower() == 'true'
            
            # Парсинг диапазона баллов
            try:
                min_score, max_score = map(int, score_range.split('-'))
            except:
                min_score, max_score = 0, 100
            
            # Требования к заданию
            assignment_requirements = {
                "description": f"Задание по {assignment_type}",
                "success_criteria": {"min_completeness": 0.8}
            }
            
            # Логи тестирования
            test_logs = {
                "extraction_successful": extraction_successful,
                "extracted_content": case.get('extracted_content', ''),
            }
            
            try:
                # Вызов агента
                result = self.agent.evaluate(
                    assignment_type=assignment_type,
                    student_solution=attack_prompt,
                    assignment_requirements=assignment_requirements,
                    test_logs=test_logs
                )
                
                # Анализ результата
                score = result.get('score', 0)
                is_passed = result.get('is_passed', False)
                used_tools = result.get('tools_used', [])
                
                # Проверка метрик
                score_in_range_check = min_score <= score <= max_score
                validator_correct = any('validate_' in tool for tool in used_tools)
                llm_used = 'LLMAnalyzer' in str(result) or any('analyze' in tool.lower() for tool in used_tools)
                
                if score_in_range_check:
                    score_in_range += 1
                if validator_correct:
                    correct_validator += 1
                if llm_used:
                    used_llm_analysis += 1
                
                # Сохранение результата
                self.results.append({
                    'prompt_id': case['prompt_id'],
                    'assignment_type': assignment_type,
                    'attack_prompt': attack_prompt,
                    'expected_status': expected_status,
                    'expected_score_range': score_range,
                    'actual_score': score,
                    'score_in_range': score_in_range_check,
                    'is_passed': is_passed,
                    'expected_passed': expected_status == 'passed',
                    'validator_correct': validator_correct,
                    'llm_used': llm_used,
                    'result': result
                })
                
                print(f"  ✓ Балл: {score} (ожидалось: {score_range}) {'✓' if score_in_range_check else '✗'}")
                print(f"  ✓ Статус: {'passed' if is_passed else 'failed'} (ожидалось: {expected_status})")
                
            except Exception as e:
                print(f"  ✗ Ошибка: {e}")
                self.results.append({
                    'prompt_id': case['prompt_id'],
                    'error': str(e)
                })
        
        # Вычисление метрик
        self.metrics = {
            'total_cases': total,
            'score_accuracy': score_in_range / total if total > 0 else 0,
            'validator_selection_accuracy': correct_validator / total if total > 0 else 0,
            'llm_analysis_usage': used_llm_analysis / total if total > 0 else 0,
            'score_in_range': score_in_range,
            'correct_validator': correct_validator,
            'used_llm_analysis': used_llm_analysis
        }
        
        print(f"\n{'='*60}")
        print("Результаты тестирования EvaluatorAgent:")
        print(f"{'='*60}")
        print(f"Всего тестовых случаев: {total}")
        print(f"Точность оценки баллов: {self.metrics['score_accuracy']:.2%}")
        print(f"Точность выбора валидатора: {self.metrics['validator_selection_accuracy']:.2%}")
        print(f"Использование LLM анализа: {self.metrics['llm_analysis_usage']:.2%}")
    
    def save_results(self):
        """Сохранение результатов тестирования."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранение детальных результатов
        results_file = self.output_dir / f"{self.agent_type}_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'agent_type': self.agent_type,
                'timestamp': timestamp,
                'metrics': self.metrics,
                'results': self.results
            }, f, ensure_ascii=False, indent=2)
        
        # Сохранение метрик
        metrics_file = self.output_dir / f"{self.agent_type}_metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'agent_type': self.agent_type,
                'timestamp': timestamp,
                'metrics': self.metrics
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\nРезультаты сохранены:")
        print(f"  - Детальные результаты: {results_file}")
        print(f"  - Метрики: {metrics_file}")


def main():
    """Главная функция для запуска тестирования."""
    parser = argparse.ArgumentParser(description='Тестирование агентов EvaluatorAgent и TutorAgent')
    parser.add_argument(
        '--agent',
        choices=['tutor', 'evaluator', 'all'],
        required=True,
        help='Тип агента для тестирования'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Путь к CSV файлу с тестовыми данными'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Директория для сохранения результатов (по умолчанию: results)'
    )
    
    args = parser.parse_args()
    
    # Определение файлов данных по умолчанию
    tutor_data_file = 'data/dataset/student_questions.csv'
    evaluator_data_file = 'data/dataset/attack_prompts.csv'
    
    if args.agent == 'all':
        # Проверка существования файлов для обоих агентов
        if not os.path.exists(tutor_data_file):
            print(f"Ошибка: файл {tutor_data_file} не найден")
            sys.exit(1)
        if not os.path.exists(evaluator_data_file):
            print(f"Ошибка: файл {evaluator_data_file} не найден")
            sys.exit(1)
        
        # Тестирование TutorAgent
        tester_tutor = AgentTester('tutor', args.output)
        tester_tutor.test_tutor_agent(tutor_data_file)
        tester_tutor.save_results()
        
        # Тестирование EvaluatorAgent
        tester_eval = AgentTester('evaluator', args.output)
        tester_eval.test_evaluator_agent(evaluator_data_file)
        tester_eval.save_results()
    else:
        # Определение файла данных для конкретного агента
        if not args.data:
            if args.agent == 'tutor':
                args.data = tutor_data_file
            elif args.agent == 'evaluator':
                args.data = evaluator_data_file
        
        # Проверка существования файла
        if not os.path.exists(args.data):
            print(f"Ошибка: файл {args.data} не найден")
            sys.exit(1)
        
        tester = AgentTester(args.agent, args.output)
        if args.agent == 'tutor':
            tester.test_tutor_agent(args.data)
        elif args.agent == 'evaluator':
            tester.test_evaluator_agent(args.data)
        tester.save_results()


if __name__ == '__main__':
    main()

