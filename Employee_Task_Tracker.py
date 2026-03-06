employees = {}


def add_employee(employee_name):
    if employee_name not in employees:
        employees[employee_name] = []
        print("employee added")
    else:
        print("employee already exist")

def assign_task(employee_name,task_name):
    if employee_name in employees:
        
        task = {
            "assignedTask" : task_name,
            "status" : "pending"
        }
        
        employees[employee_name].append(task)
        print(f"Task({task_name}) assigned to ({employee_name})")

    else:
        print("Employee not found")


def complete_task(employee_name,task_name):

    if employee_name in employees:
        for task in employees[employee_name]:
            if task["assignedTask"] == task_name:
                task["status"] = "completed"
                print("status updated")
                return            
    else:
        print("Employee Not Found")


def get_employee_tasks(employee_name):
    
    for employee in employees[employee_name]:
        print(employee)

def get_pending_tasks():

    for employee,tasks in employees.items():
        for task in tasks:
            if task["status"] == "pending":
                print(task)
        

print("--")
add_employee("waleed")
assign_task("waleed","Employee_task_tracker")
assign_task("waleed","Employee_task")
assign_task("waleed","Employee")
assign_task("waleed","Emp")
assign_task("waleed","E")
complete_task("waleed","Employee_task_tracker")
complete_task("waleed","Emp")



print("--")
add_employee("saim")
assign_task("saim","Employee_task_tracker")
assign_task("saim","Employee_task")
complete_task("saim","Employee_task_tracker")


print("--")
add_employee("asad")
assign_task("asad","Employee")
assign_task("asad","Emp")
complete_task("asad","Emp")




print("--")
get_employee_tasks("waleed")
get_employee_tasks("saim")
get_employee_tasks("asad")

print("--")
get_pending_tasks()