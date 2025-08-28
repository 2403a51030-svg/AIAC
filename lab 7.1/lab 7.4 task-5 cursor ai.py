class StudentRecord:
    def __init__(self, name, student_id, courses=None):
        self.studentName = name
        self.student_id = student_id
        self.courses = list(courses) if courses is not None else []

    def add_course(self, course):
        self.courses.append(course)

    def get_summary(self):
        courses_str = ", ".join(self.courses) if self.courses else "None"
        return f"Student: {self.studentName}, ID: {self.student_id}, Courses: {courses_str}"


class Department:
    def __init__(self, deptName, students=None):
        self.dept_name = deptName
        self.students = list(students) if students is not None else []

    def enroll_student(self, student):
        self.students.append(student)

    def department_summary(self):
        return f"Department: {self.dept_name}, Total Students: {len(self.students)}"


s1 = StudentRecord("Alice", 101, ["Math", "Science"])
d1 = Department("Computer Science")
d1.enroll_student(s1)
print(s1.get_summary())
print(d1.department_summary())