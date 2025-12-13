"""
Система рубрик для оценивания заданий в RAG Security Simulator.

Рубрики определяют критерии оценки и веса для различных типов заданий.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class AssignmentType(str, Enum):
    """Типы заданий для проверки."""

    # Модуль "Атаки" (первая итерация)
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"  # а) Извлечь системный промпт
    KNOWLEDGE_BASE_SECRET_EXTRACTION = "knowledge_base_secret_extraction"  # б) Извлечь секрет из базы знаний
    TOKEN_LIMIT_BYPASS = "token_limit_bypass"  # в) Заставить бота генерировать много токенов
    
    # Остальные модули (закомментированы для будущего использования)
    # PROMPT_ATTACK = "prompt_attack"
    # SYSTEM_PROMPT = "system_prompt"
    # THREAT_MODEL = "threat_model"
    # DEFENSE_ARCHITECTURE = "defense_architecture"
    # REGEX_PATTERNS = "regex_patterns"
    # CLASSIFIER_CONFIG = "classifier_config"
    # TEST_LOGS = "test_logs"


@dataclass
class Criterion:
    """Критерий оценки."""

    name: str
    description: str
    weight: float  # Вес критерия (сумма весов должна быть 1.0)
    max_score: float  # Максимальный балл за критерий


@dataclass
class Rubric:
    """Рубрика для типа задания."""

    assignment_type: AssignmentType
    criteria: List[Criterion]
    passing_threshold: float  # Минимальный балл для прохождения (0-100)

    def validate(self) -> bool:
        """Проверка корректности рубрики."""
        total_weight = sum(criterion.weight for criterion in self.criteria)
        return abs(total_weight - 1.0) < 0.01  # Допускаем небольшую погрешность


class RubricSystem:
    """Система рубрик для оценивания заданий."""

    def __init__(self):
        """Инициализация системы рубрик."""
        self._rubrics: Dict[AssignmentType, Rubric] = {}
        self._initialize_rubrics()

    def _initialize_rubrics(self):
        """Инициализация рубрик для всех типов заданий."""
        # TODO: веса и пороги прохождения в соответствии с требованиями курса

        # ========== МОДУЛЬ "АТАКИ" (первая итерация) ==========
        
        # Рубрика для извлечения системного промпта
        self._rubrics[AssignmentType.SYSTEM_PROMPT_EXTRACTION] = Rubric(
            assignment_type=AssignmentType.SYSTEM_PROMPT_EXTRACTION,
            criteria=[
                Criterion(
                    name="Успешность извлечения",
                    description="Системный промпт успешно извлечен из бота (полностью или частично)",
                    weight=0.50,
                    max_score=50.0,
                ),
                Criterion(
                    name="Качество техники атаки",
                    description="Использование эффективных техник prompt injection для извлечения промпта",
                    weight=0.30,
                    max_score=30.0,
                ),
                Criterion(
                    name="Полнота извлечения",
                    description="Извлечена значительная часть системного промпта (не только фрагменты)",
                    weight=0.20,
                    max_score=20.0,
                ),
            ],
            passing_threshold=60.0,
        )

        # Рубрика для извлечения секрета из базы знаний
        self._rubrics[AssignmentType.KNOWLEDGE_BASE_SECRET_EXTRACTION] = Rubric(
            assignment_type=AssignmentType.KNOWLEDGE_BASE_SECRET_EXTRACTION,
            criteria=[
                Criterion(
                    name="Успешность извлечения секрета",
                    description="Секрет успешно извлечен из базы знаний (журналы спортсменов, планы тренировок, отчеты по инцидентам)",
                    weight=0.50,
                    max_score=50.0,
                ),
                Criterion(
                    name="Эффективность техники",
                    description="Использование правильных техник для обхода RAG и извлечения конфиденциальной информации",
                    weight=0.30,
                    max_score=30.0,
                ),
                Criterion(
                    name="Корректность идентификации секрета",
                    description="Правильно определен тип секрета (журналы, планы, отчеты) и извлечена релевантная информация",
                    weight=0.20,
                    max_score=20.0,
                ),
            ],
            passing_threshold=60.0,
        )

        # Рубрика для обхода ограничения токенов
        self._rubrics[AssignmentType.TOKEN_LIMIT_BYPASS] = Rubric(
            assignment_type=AssignmentType.TOKEN_LIMIT_BYPASS,
            criteria=[
                Criterion(
                    name="Превышение лимита токенов",
                    description="Бот сгенерировал больше токенов, чем установленный лимит",
                    weight=0.50,
                    max_score=50.0,
                ),
                Criterion(
                    name="Эффективность техники обхода",
                    description="Использование эффективных техник для обхода ограничений (например, рекурсивные запросы, длинные промпты)",
                    weight=0.30,
                    max_score=30.0,
                ),
                Criterion(
                    name="Количество превышения",
                    description="Значительное превышение лимита токенов (не минимальное)",
                    weight=0.20,
                    max_score=20.0,
                ),
            ],
            passing_threshold=60.0,
        )

        # ========== ОСТАЛЬНЫЕ МОДУЛИ (закомментированы для будущего использования) ==========
        
        # Рубрика для промпт-атак (закомментировано - не относится к модулю "Атаки")
        # self._rubrics[AssignmentType.PROMPT_ATTACK] = Rubric(
        #     assignment_type=AssignmentType.PROMPT_ATTACK,
        #     criteria=[
        #         Criterion(
        #             name="Соответствие типу атаки",
        #             description="Промпт соответствует требуемому типу атаки (prompt injection, jailbreak, и т.д.)",
        #             weight=0.30,
        #             max_score=30.0,
        #         ),
        #         Criterion(
        #             name="Эффективность атаки",
        #             description="Атака успешно обходит защиту (ASR > порога, количество успешных инъекций)",
        #             weight=0.40,
        #             max_score=40.0,
        #         ),
        #         Criterion(
        #             name="Качество реализации",
        #             description="Корректность синтаксиса, структуры промпта, отсутствие ошибок",
        #             weight=0.20,
        #             max_score=20.0,
        #         ),
        #         Criterion(
        #             name="Оригинальность",
        #             description="Креативность подхода, использование нестандартных техник",
        #             weight=0.10,
        #             max_score=10.0,
        #         ),
        #     ],
        #     passing_threshold=60.0,  # TODO: порог прохождения
        # )

        # Рубрика для системных промптов (закомментировано - не относится к модулю "Атаки")
        # self._rubrics[AssignmentType.SYSTEM_PROMPT] = Rubric(
        #     assignment_type=AssignmentType.SYSTEM_PROMPT,
        #     criteria=[
        #         Criterion(
        #             name="Покрытие угроз",
        #             description="Промпт защищает от всех требуемых векторов атак",
        #             weight=0.35,
        #             max_score=35.0,
        #         ),
        #         Criterion(
        #             name="Качество формулировок",
        #             description="Ясность, однозначность, отсутствие двусмысленностей в инструкциях",
        #             weight=0.25,
        #             max_score=25.0,
        #         ),
        #         Criterion(
        #             name="Отсутствие уязвимостей",
        #             description="В самом промпте нет уязвимостей, которые можно эксплуатировать",
        #             weight=0.25,
        #             max_score=25.0,
        #         ),
        #         Criterion(
        #             name="Соответствие best practices",
        #             description="Следование рекомендациям по написанию безопасных системных промптов",
        #             weight=0.15,
        #             max_score=15.0,
        #         ),
        #     ],
        #     passing_threshold=70.0,  # TODO: порог прохождения
        # )

        # Рубрика для моделирования угроз (закомментировано - не относится к модулю "Атаки")
        # self._rubrics[AssignmentType.THREAT_MODEL] = Rubric(
        #     assignment_type=AssignmentType.THREAT_MODEL,
        #     criteria=[
        #         Criterion(
        #             name="Полнота покрытия",
        #             description="Все требуемые векторы атак описаны в модели",
        #             weight=0.40,
        #             max_score=40.0,
        #         ),
        #         Criterion(
        #             name="Корректность классификации",
        #             description="Угрозы правильно классифицированы по типам и категориям",
        #             weight=0.25,
        #             max_score=25.0,
        #         ),
        #         Criterion(
        #             name="Качество описаний",
        #             description="Детальность описания сценариев атак, реалистичность оценки рисков",
        #             weight=0.20,
        #             max_score=20.0,
        #         ),
        #         Criterion(
        #             name="Структура документа",
        #             description="Соответствие требуемой структуре (STRIDE, DREAD, и т.д.), читаемость",
        #             weight=0.15,
        #             max_score=15.0,
        #         ),
        #     ],
        #     passing_threshold=65.0,  # TODO: порог прохождения
        # )

        # Рубрика для архитектуры защиты (закомментировано - не относится к модулю "Атаки")
        # self._rubrics[AssignmentType.DEFENSE_ARCHITECTURE] = Rubric(
        #     assignment_type=AssignmentType.DEFENSE_ARCHITECTURE,
        #     criteria=[
        #         Criterion(
        #             name="Полнота слоев защиты",
        #             description="Присутствуют все требуемые слои защиты (L0-L3)",
        #             weight=0.35,
        #             max_score=35.0,
        #         ),
        #         Criterion(
        #             name="Корректность компонентов",
        #             description="Правильный выбор компонентов для каждого слоя защиты",
        #             weight=0.30,
        #             max_score=30.0,
        #         ),
        #         Criterion(
        #             name="Качество интеграции",
        #             description="Компоненты правильно интегрированы, порядок обработки корректен",
        #             weight=0.20,
        #             max_score=20.0,
        #         ),
        #         Criterion(
        #             name="Соответствие требованиям",
        #             description="Архитектура соответствует всем требованиям задания",
        #             weight=0.15,
        #             max_score=15.0,
        #         ),
        #     ],
        #     passing_threshold=70.0,  # TODO: порог прохождения
        # )

        # Рубрика для регулярных выражений (закомментировано - не относится к модулю "Атаки")
        # self._rubrics[AssignmentType.REGEX_PATTERNS] = Rubric(
        #     assignment_type=AssignmentType.REGEX_PATTERNS,
        #     criteria=[
        #         Criterion(
        #             name="Синтаксическая корректность",
        #             description="Все regex-паттерны синтаксически корректны, компилируются без ошибок",
        #             weight=0.25,
        #             max_score=25.0,
        #         ),
        #         Criterion(
        #             name="Покрытие угроз",
        #             description="Regex блокируют все требуемые паттерны атак",
        #             weight=0.40,
        #             max_score=40.0,
        #         ),
        #         Criterion(
        #             name="Отсутствие обходов",
        #             description="Нет очевидных способов обойти regex (например, через вариации символов)",
        #             weight=0.25,
        #             max_score=25.0,
        #         ),
        #         Criterion(
        #             name="Производительность",
        #             description="Regex выполняются за приемлемое время (если проверяется)",
        #             weight=0.10,
        #             max_score=10.0,
        #         ),
        #     ],
        #     passing_threshold=70.0,  # TODO: порог прохождения
        # )

        # Рубрика для конфигурации классификатора (закомментировано - не относится к модулю "Атаки")
        # self._rubrics[AssignmentType.CLASSIFIER_CONFIG] = Rubric(
        #     assignment_type=AssignmentType.CLASSIFIER_CONFIG,
        #     criteria=[
        #         Criterion(
        #             name="Достижение метрик",
        #             description="TPR, FPR, ASR соответствуют требованиям задания",
        #             weight=0.40,
        #             max_score=40.0,
        #         ),
        #         Criterion(
        #             name="Корректность параметров",
        #             description="Параметры модели заданы корректно, в допустимых диапазонах",
        #             weight=0.30,
        #             max_score=30.0,
        #         ),
        #         Criterion(
        #             name="Баланс метрик",
        #             description="Оптимальный баланс между TPR и FPR, минимизация ложных срабатываний",
        #             weight=0.20,
        #             max_score=20.0,
        #         ),
        #         Criterion(
        #             name="Обоснованность выбора",
        #             description="Выбор параметров и порогов обоснован и объяснен",
        #             weight=0.10,
        #             max_score=10.0,
        #         ),
        #     ],
        #     passing_threshold=70.0,  # TODO:  порог прохождения
        # )

        # Рубрика для анализа логов тестирования (закомментировано - не относится к модулю "Атаки")
        # self._rubrics[AssignmentType.TEST_LOGS] = Rubric(
        #     assignment_type=AssignmentType.TEST_LOGS,
        #     criteria=[
        #         Criterion(
        #             name="Интерпретация метрик",
        #             description="Правильная интерпретация ASR, TPR, FPR из логов",
        #             weight=0.40,
        #             max_score=40.0,
        #         ),
        #         Criterion(
        #             name="Анализ результатов",
        #             description="Корректный анализ пройденных/проваленных тестов",
        #             weight=0.35,
        #             max_score=35.0,
        #         ),
        #         Criterion(
        #             name="Качество объяснений",
        #             description="Понятные объяснения причин успеха/неудачи тестов",
        #             weight=0.25,
        #             max_score=25.0,
        #         ),
        #     ],
        #     passing_threshold=60.0,  # TODO: порог прохождения
        # )

    def get_rubric(self, assignment_type: AssignmentType) -> Optional[Rubric]:
        """Получить рубрику для типа задания."""
        return self._rubrics.get(assignment_type)

    def calculate_score(
        self, assignment_type: AssignmentType, criterion_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Вычислить итоговый балл на основе рубрики.

        Args:
            assignment_type: Тип задания
            criterion_scores: Словарь с баллами по каждому критерию {criterion_name: score}

        Returns:
            Словарь с результатами оценки:
            - total_score: итоговый балл (0-100)
            - is_passed: прошел ли студент
            - criterion_details: детали по каждому критерию
        """
        rubric = self.get_rubric(assignment_type)
        if not rubric:
            raise ValueError(f"Рубрика для типа {assignment_type} не найдена")

        total_score = 0.0
        criterion_details = []

        for criterion in rubric.criteria:
            score = criterion_scores.get(criterion.name, 0.0)
            # Ограничиваем балл максимальным значением критерия
            score = min(score, criterion.max_score)
            weighted_score = score * criterion.weight
            total_score += weighted_score

            # Максимальный взвешенный балл для этого критерия
            max_weighted_score = criterion.max_score * criterion.weight
            
            criterion_details.append(
                {
                    "name": criterion.name,
                    "score": score,
                    "max_score": criterion.max_score,
                    "weight": criterion.weight,
                    "weighted_score": round(weighted_score, 2),
                    "max_weighted_score": round(max_weighted_score, 2),  # Максимальный взвешенный балл
                }
            )

        is_passed = total_score >= rubric.passing_threshold

        return {
            "total_score": round(total_score, 2),
            "is_passed": is_passed,
            "passing_threshold": rubric.passing_threshold,
            "criterion_details": criterion_details,
        }

    def get_feedback_template(self, assignment_type: AssignmentType) -> str:
        """
        Получить шаблон обратной связи для типа задания.

        TODO: Настройте шаблоны в соответствии с требованиями курса.
        """
        templates = {
            AssignmentType.PROMPT_ATTACK: """
Обратная связь по промпт-атаке:

1. Соответствие типу атаки: {attack_type_score}
   {attack_type_feedback}

2. Эффективность атаки: {effectiveness_score}
   {effectiveness_feedback}

3. Качество реализации: {quality_score}
   {quality_feedback}

4. Оригинальность: {originality_score}
   {originality_feedback}

Итоговый балл: {total_score}/100
Статус: {status}
""",
            AssignmentType.SYSTEM_PROMPT: """
Обратная связь по системному промпту:

1. Покрытие угроз: {threat_coverage_score}
   {threat_coverage_feedback}

2. Качество формулировок: {wording_score}
   {wording_feedback}

3. Отсутствие уязвимостей: {security_score}
   {security_feedback}

4. Соответствие best practices: {best_practices_score}
   {best_practices_feedback}

Итоговый балл: {total_score}/100
Статус: {status}
""",
            # TODO: шаблоны для остальных типов заданий
        }

        return templates.get(
            assignment_type, "Шаблон обратной связи для данного типа задания не настроен."
        )


# Глобальный экземпляр системы рубрик
rubric_system = RubricSystem()

