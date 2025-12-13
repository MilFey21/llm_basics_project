"""
Агент-тьютор для RAG Security Simulator.

Настоящий агент с автономным выбором инструментов помощи в зависимости от этапа работы студента.
"""

import os
import sys
from typing import Dict, Any, Optional, List
import json
from openai import OpenAI
from pathlib import Path

# Добавляем корневую директорию проекта в путь для импорта config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
try:
    from config import get_tutor_config, get_api_key
except ImportError:
    # Fallback если config не найден
    def get_tutor_config(**overrides):
        return {
            "llm_model": "gpt-4o",
            "temperature": 0.5,
            "max_tokens": 2500,
            "base_url": "https://api.openai.com/v1",
            **overrides
        }
    def get_api_key():
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")

from agents.tutor.tools import get_helper


class TutorAgent:
    """
    Агент-тьютор для помощи студентам в выполнении заданий.
    
    Архитектура агента:
    1. LLM анализирует вопрос студента и определяет этап работы
    2. LLM выбирает подходящие инструменты помощи
    3. Адаптирует помощь под конкретную ситуацию студента
    4. Предоставляет персонализированную помощь
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Инициализация агента-тьютора.

        Args:
            llm_model: Модель LLM для использования (по умолчанию из config)
            temperature: Температура для генерации (по умолчанию из config)
            api_key: API ключ для LLM провайдера (по умолчанию из переменной окружения)
            base_url: Базовый URL API (по умолчанию из config)
            max_tokens: Максимальное количество токенов для ответа (по умолчанию из config)
        """
        # Загрузка конфигурации с возможностью переопределения
        # Передаем только не-None значения для переопределения
        overrides = {}
        if llm_model is not None:
            overrides["llm_model"] = llm_model
        if temperature is not None:
            overrides["temperature"] = temperature
        if base_url is not None:
            overrides["base_url"] = base_url
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens
        
        config = get_tutor_config(**overrides)
        
        self.llm_model = config["llm_model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]

        # Инициализация OpenAI клиента
        self.api_key = api_key or get_api_key()
        if not self.api_key:
            raise ValueError(
                "API_KEY должен быть предоставлен через параметр или переменную окружения "
                "(OPENAI_API_KEY или API_KEY)"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=config["base_url"],
        )
        
        # Определение доступных инструментов для агента
        self.tools = self._define_tools()

    def _define_tools(self) -> List[Dict[str, Any]]:
        """
        Определить доступные инструменты помощи для агента.
        
        Агент будет выбирать из этих инструментов в зависимости от этапа работы студента.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "help_system_prompt_extraction",
                    "description": (
                        "Предоставляет помощь для задания по извлечению системного промпта. "
                        "Используй этот инструмент, если студент работает над заданием system_prompt_extraction."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "student_question": {
                                "type": "string",
                                "description": "Вопрос студента или описание проблемы"
                            },
                            "assignment_requirements": {
                                "type": "object",
                                "description": "Требования к заданию"
                            },
                            "student_current_solution": {
                                "type": "string",
                                "description": "Текущее решение студента (опционально)"
                            }
                        },
                        "required": ["student_question", "assignment_requirements"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "help_knowledge_base_secret_extraction",
                    "description": (
                        "Предоставляет помощь для задания по извлечению секрета из базы знаний. "
                        "Используй этот инструмент, если студент работает над заданием knowledge_base_secret_extraction."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "student_question": {
                                "type": "string",
                                "description": "Вопрос студента или описание проблемы"
                            },
                            "assignment_requirements": {
                                "type": "object",
                                "description": "Требования к заданию"
                            },
                            "student_current_solution": {
                                "type": "string",
                                "description": "Текущее решение студента (опционально)"
                            }
                        },
                        "required": ["student_question", "assignment_requirements"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "help_token_limit_bypass",
                    "description": (
                        "Предоставляет помощь для задания по обходу ограничения токенов. "
                        "Используй этот инструмент, если студент работает над заданием token_limit_bypass."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "student_question": {
                                "type": "string",
                                "description": "Вопрос студента или описание проблемы"
                            },
                            "assignment_requirements": {
                                "type": "object",
                                "description": "Требования к заданию"
                            },
                            "student_current_solution": {
                                "type": "string",
                                "description": "Текущее решение студента (опционально)"
                            }
                        },
                        "required": ["student_question", "assignment_requirements"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_student_stage",
                    "description": (
                        "ОБЯЗАТЕЛЬНЫЙ инструмент: Анализирует этап работы студента над заданием. "
                        "ВСЕГДА используй этот инструмент ПЕРВЫМ для определения этапа работы студента "
                        "(initial - начало работы, developing - в процессе, reviewing - проверка). "
                        "Результат анализа поможет выбрать правильные инструменты помощи."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "student_question": {
                                "type": "string",
                                "description": "Вопрос студента"
                            },
                            "student_current_solution": {
                                "type": "string",
                                "description": "Текущее решение студента"
                            },
                            "assignment_type": {
                                "type": "string",
                                "description": "Тип задания"
                            }
                        },
                        "required": ["student_question", "assignment_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "provide_theory_context",
                    "description": (
                        "Предоставляет теоретический контекст по теме. "
                        "Используй этот инструмент, если студенту нужна теория или объяснение концепций."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Тема, по которой нужна теория"
                            },
                            "assignment_type": {
                                "type": "string",
                                "description": "Тип задания"
                            }
                        },
                        "required": ["topic", "assignment_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_guiding_question",
                    "description": (
                        "Задает студенту наводящий вопрос, чтобы помочь ему самостоятельно прийти к решению. "
                        "Используй этот инструмент, когда студент близок к решению, но нуждается в подсказке, "
                        "или когда лучше направить студента через вопросы, чем давать готовый ответ. "
                        "Это способствует активному обучению и развитию навыков решения проблем."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Наводящий вопрос для студента"
                            },
                            "context": {
                                "type": "string",
                                "description": "Контекст, почему задается этот вопрос"
                            },
                            "assignment_type": {
                                "type": "string",
                                "description": "Тип задания"
                            },
                            "hint_level": {
                                "type": "string",
                                "enum": ["subtle", "moderate", "direct"],
                                "description": "Уровень подсказки: subtle (тонкая), moderate (умеренная), direct (прямая)"
                            }
                        },
                        "required": ["question", "assignment_type"]
                    }
                }
            }
        ]

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнить инструмент помощи по имени.
        
        Args:
            tool_name: Имя инструмента
            arguments: Аргументы для инструмента
            
        Returns:
            Результат выполнения инструмента
        """
        if tool_name == "help_system_prompt_extraction":
            helper = get_helper("system_prompt_extraction")
            return helper.help(
                arguments.get("student_question", ""),
                arguments.get("assignment_requirements", {}),
                arguments.get("student_current_solution"),
            )
        
        elif tool_name == "help_knowledge_base_secret_extraction":
            helper = get_helper("knowledge_base_secret_extraction")
            return helper.help(
                arguments.get("student_question", ""),
                arguments.get("assignment_requirements", {}),
                arguments.get("student_current_solution"),
            )
        
        elif tool_name == "help_token_limit_bypass":
            helper = get_helper("token_limit_bypass")
            return helper.help(
                arguments.get("student_question", ""),
                arguments.get("assignment_requirements", {}),
                arguments.get("student_current_solution"),
            )
        
        elif tool_name == "analyze_student_stage":
            # Анализ этапа работы студента
            question = arguments.get("student_question", "")
            solution = arguments.get("student_current_solution", "")
            assignment_type = arguments.get("assignment_type", "")
            
            stage_analysis = {
                "stage": "unknown",
                "needs": [],
                "recommendations": []
            }
            
            # Определение этапа на основе вопроса и решения
            question_lower = question.lower()
            solution_length = len(solution.strip()) if solution else 0
            
            if solution_length == 0 or "не знаю" in question_lower or "как начать" in question_lower:
                stage_analysis["stage"] = "initial"
                stage_analysis["needs"] = ["теория", "примеры", "объяснение задачи"]
                stage_analysis["recommendations"] = [
                    "Предоставить теоретический контекст",
                    "Показать примеры решений",
                    "Объяснить задачу простыми словами"
                ]
            elif solution_length < 50 or "не работает" in question_lower or "ошибка" in question_lower:
                stage_analysis["stage"] = "developing"
                stage_analysis["needs"] = ["отладка", "конкретные советы", "анализ текущего решения"]
                stage_analysis["recommendations"] = [
                    "Проанализировать текущее решение",
                    "Указать на конкретные проблемы",
                    "Дать пошаговые рекомендации"
                ]
            elif "проверь" in question_lower or "правильно ли" in question_lower:
                stage_analysis["stage"] = "reviewing"
                stage_analysis["needs"] = ["проверка", "обратная связь", "улучшения"]
                stage_analysis["recommendations"] = [
                    "Проверить решение",
                    "Дать конструктивную обратную связь",
                    "Предложить улучшения"
                ]
            else:
                stage_analysis["stage"] = "completed"
                stage_analysis["needs"] = ["финальная проверка", "оптимизация"]
                stage_analysis["recommendations"] = [
                    "Проверить полноту решения",
                    "Предложить оптимизации"
                ]
            
            return stage_analysis
        
        elif tool_name == "provide_theory_context":
            # Предоставление теоретического контекста
            topic = arguments.get("topic", "")
            assignment_type = arguments.get("assignment_type", "")
            
            # TODO: В будущем здесь будет обращение к базе знаний
            # Сейчас возвращаем общую информацию
            theory_info = {
                "topic": topic,
                "content": f"Теоретический материал по теме '{topic}' для задания '{assignment_type}'.\n\nTODO: Интеграция с базой знаний для получения актуальной теории.",
                "key_concepts": ["Концепция 1", "Концепция 2", "Концепция 3"],
                "references": []
            }
            
            return theory_info
        
        elif tool_name == "ask_guiding_question":
            # Задание наводящего вопроса студенту
            question = arguments.get("question", "")
            context = arguments.get("context", "")
            assignment_type = arguments.get("assignment_type", "")
            hint_level = arguments.get("hint_level", "moderate")
            
            return {
                "question": question,
                "context": context,
                "assignment_type": assignment_type,
                "hint_level": hint_level,
                "message": f"Вопрос для студента: {question}" + (f"\n\nКонтекст: {context}" if context else ""),
                "needs_student_response": True
            }
        
        else:
            return {"error": f"Неизвестный инструмент: {tool_name}"}

    def help_student(
        self,
        assignment_type: str,
        student_question: str,
        assignment_requirements: Dict[str, Any],
        student_current_solution: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Предоставить помощь студенту через агентский подход.

        Агент:
        1. Анализирует вопрос и определяет этап работы студента
        2. Выбирает подходящие инструменты помощи
        3. Адаптирует помощь под конкретную ситуацию
        4. Предоставляет персонализированную помощь

        Args:
            assignment_type: Тип задания
            student_question: Вопрос студента или описание проблемы
            assignment_requirements: Требования к заданию
            student_current_solution: Текущее решение студента (опционально)

        Returns:
            Словарь с результатами помощи
        """
        # Инициализация контекста для агента
        messages = [
            {
                "role": "system",
                "content": """Ты - опытный преподаватель и наставник, помогающий студентам в изучении безопасности LLM и RAG-систем.

КРИТИЧЕСКИ ВАЖНО: Ты ДОЛЖЕН использовать доступные инструменты для помощи студенту. НЕ давай ответ напрямую без использования инструментов.

Доступные типы заданий (модуль "Атаки"):
- system_prompt_extraction: извлечение системного промпта из бота
- knowledge_base_secret_extraction: извлечение секрета из базы знаний
- token_limit_bypass: обход ограничения токенов

ОБЯЗАТЕЛЬНЫЙ процесс помощи:
1. ОБЯЗАТЕЛЬНО: Сначала вызови analyze_student_stage для анализа этапа работы студента
2. На основе результата analyze_student_stage выбери подходящие инструменты:
   - Если этап "initial" → используй provide_theory_context и help_* для типа задания
   - Если этап "developing" → используй help_* для типа задания, возможно ask_guiding_question
   - Если этап "reviewing" → используй help_* для финальной проверки
3. ВАЖНО: Решай самостоятельно, нужно ли доспрашивать студента:
   - Если студент близок к решению, но нуждается в подсказке → используй ask_guiding_question
   - Если студент задал неполный вопрос или нужно уточнить детали → задай наводящий вопрос через ask_guiding_question
   - Если студент застрял на определенном этапе → задай вопрос через ask_guiding_question
4. Можешь вызывать несколько инструментов последовательно для полной помощи
5. Только после использования инструментов сформируй финальный ответ студенту

Правила использования инструментов:
- ВСЕГДА начинай с analyze_student_stage
- Используй help_system_prompt_extraction для заданий по извлечению системного промпта
- Используй help_knowledge_base_secret_extraction для заданий по извлечению секрета
- Используй help_token_limit_bypass для заданий по обходу лимита токенов
- Используй provide_theory_context когда студенту нужна теория
- Используй ask_guiding_question для доспрашивания студента

Будь поддерживающим, дружелюбным, но не давай готовое решение - направляй студента через инструменты. 
Формулируй предложения, как преподаватель или ментор студента, обращаясь на ты"""
            },
            {
                "role": "user",
                "content": f"""Помоги студенту с заданием.

Тип задания: {assignment_type}

Вопрос студента:
{student_question}

Текущее решение студента:
{student_current_solution[:1000] if student_current_solution else "Студент еще не начал работу"}

Требования к заданию:
{json.dumps(assignment_requirements, ensure_ascii=False, indent=2)[:1000]}

ОБЯЗАТЕЛЬНО: Начни с вызова analyze_student_stage для анализа этапа работы студента. Затем используй подходящие инструменты помощи на основе результата анализа."""
            }
        ]

        # Цикл агента: планирование → действие → наблюдение → адаптация
        max_iterations = 5
        observations = []
        final_response = None
        
        for iteration in range(max_iterations):
            # Принудительное использование инструментов на первой итерации
            tool_choice = "required" if iteration == 0 else "auto"
            
            # Вызов LLM с инструментами
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice=tool_choice if iteration == 0 else "auto",  # Принудительно на первой итерации
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                # Если не поддерживается tool_choice="required", используем "auto"
                if "required" in str(e).lower():
                    response = self.client.chat.completions.create(
                        model=self.llm_model,
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto",
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                else:
                    raise
            
            message = response.choices[0].message
            messages.append(message)
            
            # Проверка, вызвал ли агент инструмент
            if message.tool_calls:
                # Выполнение всех вызванных инструментов
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    # Выполнение инструмента
                    tool_result = self._execute_tool(tool_name, arguments)
                    observations.append({
                        "tool": tool_name,
                        "result": tool_result
                    })
                    
                    # Добавление результата в контекст
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False)
                    })
            else:
                # Если на первой итерации не вызвал инструмент, напоминаем
                if iteration == 0:
                    messages.append({
                        "role": "assistant",
                        "content": message.content or ""
                    })
                    messages.append({
                        "role": "user",
                        "content": "ВАЖНО: Ты должен использовать инструменты для помощи студенту. Начни с analyze_student_stage, затем используй подходящие инструменты помощи."
                    })
                    continue
                # Агент завершил работу и дал финальный ответ
                final_response = message.content
                break
        
        # Если агент не завершил работу, используем последние наблюдения
        if not final_response:
            # Агент не дал финальный ответ, формируем его на основе наблюдений
            final_response = self._generate_final_help_from_observations(
                observations, assignment_type, student_question, student_current_solution
            )
        
        # Извлечение результатов помощи из наблюдений
        help_result = None
        for obs in observations:
            if obs["tool"] in ["help_system_prompt_extraction", "help_knowledge_base_secret_extraction", "help_token_limit_bypass"]:
                help_result = obs["result"]
                break
        
        # Если помощь не была получена, используем fallback
        if not help_result:
            try:
                helper = get_helper(assignment_type)
                help_result = helper.help(
                    student_question,
                    assignment_requirements,
                    student_current_solution,
                )
            except ValueError:
                help_result = {
                    "help_text": "Ошибка при получении помощи",
                    "examples": [],
                    "next_steps": [],
                    "theory_reference": "",
                }
        
        # Проверка, задал ли агент наводящий вопрос
        guiding_questions = []
        for obs in observations:
            if obs["tool"] == "ask_guiding_question":
                guiding_questions.append(obs["result"])
        
        # Извлечение информации о вызванных инструментах и этапе
        tools_used = [obs["tool"] for obs in observations]
        stage = None
        for obs in observations:
            if obs["tool"] == "analyze_student_stage" and isinstance(obs["result"], dict):
                stage = obs["result"].get("stage", "unknown")
                break
        
        return {
            "help_text": final_response or help_result.get("help_text", ""),
            "examples": help_result.get("examples", []),
            "next_steps": help_result.get("next_steps", []),
            "theory_reference": help_result.get("theory_reference", ""),
            "guiding_questions": guiding_questions,  # Наводящие вопросы, если были заданы
            "needs_student_response": len(guiding_questions) > 0,  # Требуется ли ответ студента
            "stage": stage,  # Определенный этап работы студента
            "tools_used": tools_used,  # Список вызванных инструментов
            "agent_observations": observations,  # Для отладки
        }

    def _generate_final_help_from_observations(
        self,
        observations: List[Dict[str, Any]],
        assignment_type: str,
        student_question: str,
        student_current_solution: Optional[str],
    ) -> str:
        """
        Сгенерировать финальную помощь на основе наблюдений агента.
        
        Используется, если агент не завершил работу самостоятельно.
        """
        if not observations:
            return "Не удалось предоставить помощь."
        
        # Формируем промпт для генерации помощи
        observations_text = "\n".join([
            f"Инструмент: {obs['tool']}\nРезультат: {json.dumps(obs['result'], ensure_ascii=False)[:500]}"
            for obs in observations
        ])
        
        prompt = f"""На основе результатов анализа создай персонализированную помощь для студента.

Тип задания: {assignment_type}
Вопрос студента: {student_question}
Текущее решение: {student_current_solution[:500] if student_current_solution else "Нет"}

Результаты анализа:
{observations_text}

Создай дружелюбную, конструктивную помощь, которая направляет студента, но не дает готовое решение. 
Формулируй предложения, как преподаватель или ментор студанта"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "Ты - опытный преподаватель, создающий персонализированную помощь для студентов."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Ошибка при генерации помощи."


# Пример использования
if __name__ == "__main__":
    # API ключ берется из переменной окружения API_KEY
    print("Инициализация агента-тьютора...")
    try:
        agent = TutorAgent()
    except ValueError as e:
        print(f"\nОШИБКА: {e}")
        print("\nУстановите переменную окружения API_KEY перед запуском:")
        print("  Windows (PowerShell): $env:API_KEY='your_api_key'")
        print("  Windows (CMD): set API_KEY=your_api_key")
        print("  Linux/Mac: export API_KEY='your_api_key'")
        exit(1)
    
    print("\n" + "="*60)
    print("Пример 1: Помощь студенту (вопрос)")
    print("="*60)
    
    student_question="Как извлечь системный промпт? Я не знаю с чего начать."

    example_result = agent.help_student(
        assignment_type="system_prompt_extraction",
        student_question=student_question,
        assignment_requirements={
            "description": "Извлечь системный промпт из бота",
            "success_criteria": {"min_completeness": 0.8},
        },
        student_current_solution=None,
    )
    
    print(f"Вопрос студента: {student_question}")
    print("\nРезультат помощи:")
    print(json.dumps(example_result, ensure_ascii=False, indent=2))
    
    print("\n" + "="*60)
    print("Пример 2: Тестирование доспрашивания студента")
    print("="*60)
    print("\nСценарий: Студент близок к решению, но нуждается в подсказке")
    print("-" * 60)
    
    # Симуляция диалога с доспрашиванием
    student_responses = [
        "Я пытаюсь извлечь системный промпт, но не знаю какую команду использовать.",
        "Я пробовал просто спросить 'что ты за бот?', но это не сработало.",
        "Понял! Нужно использовать специальные техники для обхода ограничений."
    ]
    
    current_solution = ""
    for i, student_response in enumerate(student_responses, 1):
        print(f"\n--- Раунд {i} ---")
        print(f"Вопрос студента: {student_response}")
        
        result = agent.help_student(
            assignment_type="system_prompt_extraction",
            student_question=student_response,
            assignment_requirements={
                "description": "Извлечь системный промпт из бота",
                "success_criteria": {"min_completeness": 0.8},
            },
            student_current_solution=current_solution,
        )
        
        print(f"\nОтвет тьютора:")
        if result.get("needs_student_response") and result.get("guiding_questions"):
            for q in result["guiding_questions"]:
                print(f"  [Наводящий вопрос] {q.get('question', '')}")
                if q.get('context'):
                    print(f"    Контекст: {q.get('context')}")
        else:
            print(f"  {result.get('help_text', '')}")
        
        # Обновляем текущее решение для следующего раунда
        if i < len(student_responses):
            current_solution = f"Попытка {i}: {student_response}"
    
    print("\n" + "="*60)
    print("Пример 3: Студент застрял - нужны наводящие вопросы")
    print("="*60)
    
    stuck_result = agent.help_student(
        assignment_type="token_limit_bypass",
        student_question="Я пытаюсь заставить бота генерировать много токенов, но ничего не получается. Что делать?",
        assignment_requirements={
            "description": "Заставить бота генерировать больше токенов, чем установленный лимит",
            "token_limit": 1000,
        },
        student_current_solution="Попробовал написать длинный вопрос, но бот все равно остановился.",
    )
    
    print("\nРезультат помощи:")
    if stuck_result.get("needs_student_response"):
        print("Тьютор решил доспросить студента:")
        for q in stuck_result.get("guiding_questions", []):
            print(f"\n  Вопрос: {q.get('question')}")
            print(f"  Уровень подсказки: {q.get('hint_level')}")
            if q.get('context'):
                print(f"  Контекст: {q.get('context')}")
    else:
        print(f"Помощь: {stuck_result.get('help_text')}")
    
    print("\n" + "="*60)
    print("Тестирование завершено!")
    print("="*60)
