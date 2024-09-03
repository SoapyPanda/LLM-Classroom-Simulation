import datetime
import random
import os
import PyPDF2
from openai import OpenAI

# Set up your OpenAI API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Emotions class to influence memory and attention
class Emotion:
    def __init__(self, name, intensity):
        self.name = name
        self.intensity = intensity  # Intensity can affect memory importance

# Attention Mechanism
class Attention:
    def __init__(self, base_level=5):
        self.base_level = base_level

    def adjust_attention(self, emotion_intensity, external_cue=False):
        if external_cue:
            self.base_level += 2  # External cues can increase attention
        self.base_level += emotion_intensity  # Emotions affect attention

    def get_attention_level(self):
        return max(1, min(10, self.base_level))  # Keep attention within a reasonable range

# Memory System
class MemoryChunk:
    def __init__(self, description, importance, attention_level, timestamp=None, decay_rate=0.01):
        self.description = description
        self.importance = importance
        self.attention_level = attention_level  # New attribute to store attention level when memory was formed
        self.creation_time = timestamp or datetime.datetime.now()
        self.last_access_time = self.creation_time
        self.decay_rate = decay_rate

    def update_access_time(self):
        self.last_access_time = datetime.datetime.now()

    def decay(self):
        time_elapsed = (datetime.datetime.now() - self.last_access_time).total_seconds()
        attention_modifier = 1 / (1 + (self.attention_level - 5) * 0.1)  # Higher attention slows down decay
        decay_factor = (1 - self.decay_rate * attention_modifier) ** time_elapsed
        self.importance *= decay_factor

class MemoryStream:
    def __init__(self):
        self.short_term_memory = []
        self.long_term_memory = []

    def add_chunk_to_short_term(self, memory_chunk):
        self.short_term_memory.append(memory_chunk)

    def transfer_to_long_term(self):
        for chunk in self.short_term_memory:
            if chunk.importance > 7:  # Example threshold for importance
                self.long_term_memory.append(chunk)
        self.short_term_memory = []  # Clear short-term memory after transfer

    def retrieve_memories(self, query=None, max_results=5):
        for memory in self.long_term_memory:
            memory.decay()

        # If a query is provided, dynamically assess relevance with importance
        if query:
            prompt = (
                f"Given the query '{query}', rate the relevance of each of the following memories: "
                f"{[memory.description for memory in self.long_term_memory]}."
                f"Please output the relevance score as a number integer between 0 and 10 on it's own line with nothing else on it for each memory"
            )
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4",
                max_tokens=100
            )
            relevance_scores = list(map(int, response.choices[0].message.content.strip().split("\n")))
            
            # Combine importance and relevance scores
            combined_scores = [
                (memory, memory.importance * relevance) for memory, relevance in zip(self.long_term_memory, relevance_scores)
            ]

            # Sort by combined score (importance * relevance)
            sorted_memories = sorted(combined_scores, key=lambda x: x[1], reverse=True)
            return [mem[0] for mem in sorted_memories[:max_results]]

        # Default to sorting by importance if no query is given
        sorted_memories = sorted(
            self.long_term_memory,
            key=lambda mem: mem.importance,
            reverse=True
        )

        return sorted_memories[:max_results]

# Reflection Mechanism
class Reflection:
    def __init__(self, insight, supporting_memories):
        self.insight = insight
        self.supporting_memories = supporting_memories

# Curiosity and Error Feedback
class Curiosity:
    def __init__(self, level=5):
        self.level = level  # Higher levels of curiosity lead to more engagement

    def generate_prediction(self, topic):
        if self.level > 5:
            # Generate a prediction or question
            prompt = f"What would be a reasonable prediction or question about the topic {topic}?"
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4",
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        return None

    def evaluate_prediction(self, prediction, reality):
        # Compare prediction to reality and adjust curiosity or memory importance
        if prediction and prediction not in reality:
            return "Surprised", 2  # Higher surprise = more memorable
        return "Expected", 0  # No surprise, no additional impact

# Agent class with emotions, attention, and curiosity
class Agent:
    def __init__(self, name):
        self.name = name
        self.memory_stream = MemoryStream()
        self.attention = Attention()
        self.emotion = None
        self.curiosity = Curiosity()

    def chunk_experience(self, experience_description):
        # Use GPT-4 to chunk the experience into smaller pieces
        prompt = (
            f"""Break down the following experience into smaller, meaningful memory chunks or key points: 
            {experience_description}
            Output only the key chunks, in the form of single concise sentences, each on its own line.
            Do not number them or add anything else, just the sentence, one per line.
            Try to only take from the experience description and not hallucinate other details."""
        )
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="gpt-4",
            max_tokens=500
        )
        chunks = response.choices[0].message.content.strip().split("\n")
        print(f"\n{self.name}'s memory chunks: {[chunk.strip() for chunk in chunks if chunk.strip()]}")
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def learn(self, experience_description):
        # Generate an emotional state (randomly for now)
        self.emotion = Emotion(name="Interested", intensity=random.randint(1, 10))
        
        # Adjust attention based on emotion
        self.attention.adjust_attention(self.emotion.intensity, external_cue=True)

        # Chunk experience into memory
        chunks = self.chunk_experience(experience_description)
        for chunk in chunks:
            importance = random.randint(5, 10) + self.attention.get_attention_level()  # Adjust importance with attention
            memory_chunk = MemoryChunk(description=chunk, importance=importance, attention_level=self.attention.get_attention_level())
            self.memory_stream.add_chunk_to_short_term(memory_chunk)

        # Simulate predictions or questions based on curiosity
        prediction = self.curiosity.generate_prediction(experience_description)
        if prediction:
            print(f"\n{self.name} predicts or asks: {prediction}")

        # Simulate transfer to long-term memory after reflection
        self.reflect(prediction)

    def reflect(self, prediction=None):
        recent_chunks = self.memory_stream.short_term_memory
        reflection_text = self.generate_reflection_text(recent_chunks)

        if prediction:
            surprise_level, importance_boost = self.curiosity.evaluate_prediction(prediction, reflection_text)
            print(f"\n{self.name} feels {surprise_level} after reflection.")
            for chunk in recent_chunks:
                chunk.importance += importance_boost  # Boost memory importance if surprised

        reflection = Reflection(insight=reflection_text, supporting_memories=recent_chunks)
        self.memory_stream.add_chunk_to_short_term(MemoryChunk(description=reflection.insight, importance=8, attention_level=self.attention.get_attention_level()))
        self.memory_stream.transfer_to_long_term()

    def generate_reflection_text(self, recent_chunks):
        chunk_descriptions = [chunk.description for chunk in recent_chunks]
        prompt = (
            f"{self.name} is reflecting on the following recent memory chunks: {chunk_descriptions}. What insights or connections can be drawn?"
        )
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="gpt-4",
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    
    def give_presentation(self, topic):
        # Retrieve the most important memories related to the topic
        print(f"\n{self.name} is preparing their presentation.")
        memories = self.memory_stream.retrieve_memories(query=topic, max_results=5)

        if memories:
            memory_content = " ".join([memory.description for memory in memories])

            # Generate the presentation content using GPT-4
            prompt = (
                f"Create a short well-structured presentation based only on the following memories: {memory_content}. "
                f"Structure it with an introduction, main points, and a conclusion."
            )

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4",
                max_tokens=500
            )
            presentation_content = response.choices[0].message.content.strip()
            
            print(f"\n{self.name}'s Presentation:")
            print(presentation_content)
            return presentation_content
        else:
            print(f"{self.name}: I don't remember enough to present on {topic}.")
            return "I don't remember enough to present."

# Teacher class
class Teacher:
    def __init__(self, name):
        self.name = name

    def give_lecture(self, topic):
        prompt = f"Generate a concise, informative lecture on the topic of {topic}. Keep the lecture short, around 5-6 sentences."
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="gpt-4",
            max_tokens=500
        )
        lecture_content = response.choices[0].message.content.strip()
        return lecture_content
    
    def give_PDF_lecture(self, PDF_subject, topic):
        system_message = """You are a knowledgeable and experienced teacher. 
        Your task is to deliver a lecture on the given topic. 
        Do not break character or acknowledge that you are an AI. 
        Respond as if you are directly addressing a student in a classroom setting.
        Limit your response to at most 150 words.
        """
        prompt = f"Please deliver a lecture on the topic '{topic}' using information from the following subject material: {PDF_subject}. Structure your lecture with an introduction, main points, and a conclusion. Do not assign any homework or ask any questions to the student."

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4o-mini",
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    def provide_feedback(self, student, presentation, topic):
        prompt = (
            f"As a teacher, evaluate the following presentation given by {student.name} on the topic entire'{topic}':\n\n"
            f"Presentation: {presentation}\n\n"
            f"Please provide a grade (A+ to F-) on the student's work"
        )

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4",
            max_tokens=200
        )
        feedback = response.choices[0].message.content.strip()
        
        # Print the feedback
        print(f"\nFeedback for {student.name}:")
        print(feedback)

# Classroom Simulation
class ClassroomSimulation:
    def __init__(self, teacher, students, subject_PDF_path="None"):
        self.teacher = teacher
        self.students = students   
        self.subject_PDF_path = subject_PDF_path
        self.subject = "Hyperian Physics"

        with open(subject_PDF_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        self.subject = text

    def conduct_lecture(self, topic):
        lecture_content = self.teacher.give_PDF_lecture(self.subject, topic)
        print(f"\n{self.teacher.name}: {lecture_content}")
        for student in self.students:
            student.learn(lecture_content)

    def interact(self, agent1, agent2, topic):
        prompt = f"{agent1.name} and {agent2.name} are having a short conversation about {topic}. What would they say?"
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="gpt-4",
            max_tokens=500
        )

        # stores memory of conversation
        agent1.learn(response.choices[0].message.content.strip())
        agent2.learn(response.choices[0].message.content.strip())

        print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    
    def presentation_day(self):
        for student in self.students:
            presentation = student.give_presentation(self.subject)
            self.teacher.provide_feedback(student, presentation, self.subject)
            input("Press Enter to continue to the next interaction...\n")


# Example usage
teacher = Teacher("Mr. Smith")
student1 = Agent("Alice")
student2 = Agent("Bob")
students = [student1, student2]

classroom = ClassroomSimulation(teacher, students, "Lesson_Plan_A.pdf")

# Teacher gives a series of lectures then students present
lectures = ["lesson 1", "lesson 2", "lesson 3", "lesson 4", "lesson 5", "lesson 6", "lesson 7", "lesson 8", "lesson 9", "lesson 10"]
for lecture in lectures:
    classroom.conduct_lecture(lecture)
    input("Press Enter to continue to the next interaction...\n")
classroom.presentation_day()