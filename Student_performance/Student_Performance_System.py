import pandas as pd

def load_data():
    df = pd.read_csv('students.csv')
    return df

def save_data(df):
    df.to_csv('students.csv', index=False)

def name_check(name):
  return name.strip().replace(' ', '').lower()

def marks_check(student_id, name, math, english, science, history):
  if not (0 <= math <= 100 and 0 <= english <= 100 and 0 <= science <= 100 and 0 <= history <= 100):
        print(f"Error: Marks for student {name} (ID: {student_id}) must be between 0 and 100. Skipping addition.")
        return False
  return True

def add_Student(df, student_id, name, math, english, science, history):
    name = name_check(name)

    if not marks_check(student_id, name, math, english, science, history):
        return df

    if student_id in df['id'].values:
        print(f"Student with ID {student_id} already exists. Skipping addition.")
        return df
    new_student = {
        'id': student_id,
        'name': name,
        'math': math,
        'english': english,
        'science': science,
        'history': history
    }
    df = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True)
    save_data(df)
    print(f"Student {name} (ID: {student_id}) added successfully.")
    return df

def update_name(df, student_id, new_name):
    processed_new_name = name_check(new_name)

    if (df["id"] == student_id).any():
        old_name_in_df = df.loc[df["id"] == student_id, "name"].iloc[0]
        df.loc[df["id"] == student_id, "name"] = processed_new_name
        save_data(df)
        print(f"Updated name for student ID '{student_id}' from '{old_name_in_df}' to '{new_name}'.")
    else:
        print(f"Student with ID '{student_id}' not found.")
    return df

def marks_check_single(student_id, mark_value):
  if not (0 <= mark_value <= 100):
        print(f"Error: Marks for student (ID: {student_id}) must be between 0 and 100. Skipping update.")
        return False
  return True

def update_marks(df, student_id, subject, new_marks):
    if (df["id"] == student_id).any():
        if subject in df.columns:
            if not marks_check_single(student_id, new_marks):
              return df
            df.loc[df["id"] == student_id, subject] = new_marks
            save_data(df)
            print(f"Updated {subject} marks for student ID '{student_id}' to {new_marks}.")
        else:
            print(f"Subject '{subject}' not found. Available subjects are: math, english, science, history.")
    else:
        print(f"Student with ID '{student_id}' not found.")
    return df

def delete_student(df, student_id):
    if (df["id"] == student_id).any():
        df.drop(df[df["id"] == student_id].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        save_data(df)
        print(f"Student with ID '{student_id}' deleted.")
    else:
        print(f"Student with ID '{student_id}' not found.")
    return df

def search_student(df, name):
    name = name_check(name)
    if (df["name"] == name).any():
      found_students = df[df["name"] == name]
      print(found_students)
    else:
      print("No student found with the given name.")

def report_data(df):
    df.to_csv('StudentReport.csv', index=False)

def grade_calc(average):
    if average >= 85:
        return 'Excellent'
    elif average >= 70:
        return 'Good'
    elif average >= 50:
        return 'Average'
    else:
        return 'Poor'

def Student_report(df):
  rdf = df.copy()
  rdf["total"] = rdf["math"] + rdf["english"] + rdf["science"] + rdf["history"]
  rdf["average"] = rdf["total"] / 4
  rdf["grade"] = rdf["average"].apply(grade_calc)
  newdf = rdf[["id", "name", "average","grade"]]
  report_data(newdf)
  return newdf

def topStudent():
    report_df = pd.read_csv("StudentReport.csv")
    top_student_id = report_df.loc[report_df["average"].idxmax(), "id"]
    print(f"The top student ID is: {top_student_id}")
  
def topSub(df):
  subjects = ["math", "english", "science", "history"]
  print("Top student for each subject:")
  for subject in subjects:
    top_score_row = df.loc[df[subject].idxmax()]
    top_student_name = top_score_row['name']
    top_score = top_score_row[subject]
    print(f"  Subject: {subject}, Student: {top_student_name}, Score: {top_score}")


df = load_data()

while True:
    print("\n--- Student Management System ---")
    print("1. Add Student")
    print("2. Update Student Name")
    print("3. Update Student Marks")
    print("4. Delete Student")
    print("5. Search Student")
    print("6. Generate Student Report")
    print("7. Find Top Student (by average score)")
    print("8. Find Top Student (by subject)")
    print("9. Exit")

    choice = input("Enter your choice: ")

    if choice == '1':
        student_id = input("Enter Student ID: ")
        name = input("Enter Student Name: ")
        math = float(input("Enter Math Marks: "))
        science = float(input("Enter Science Marks: "))
        english = float(input("Enter English Marks: "))
        history = float(input("Enter History Marks: "))
        df = add_Student(df, student_id, name, math, science, english, history)
    elif choice == '2':
        student_id = input("Enter Student ID to update name: ")
        new_name = input("Enter new name: ")
        df = update_name(df, student_id, new_name)
    elif choice == '3':
        student_id = input("Enter Student ID to update marks: ")
        subject = input("Enter subject (math, english, science, history): ").lower()
        new_marks = float(input("Enter new marks: "))
        df = update_marks(df, student_id, subject, new_marks)
    elif choice == '4':
        student_id = input("Enter Student ID to delete: ")
        df = delete_student(df, student_id)
    elif choice == '5':
        name = input("Enter student name to search: ")
        search_student(df, name)

    elif choice == '6':
        report_df = Student_report(df)
        print(report_df)
    elif choice == '7':
        topStudent()
    elif choice == '8':
        topSub(df)
    elif choice == '9':
        print("Exiting Student Management System. Goodbye!")
        break

    else:
        print("Invalid option. Please choose a number between 1 and 9.")
