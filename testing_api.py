import requests

url = 'http://localhost:8081/update'

action_designator_str = """PickUpAction(object_designator=Object(name='Cup',concept='Cup', color='blue'), arm=Arms.LEFT, grasp_description=GraspDescription(approach_direction=Grasp.TOP,vertical_alignment=Grasp.TOP, rotate_gripper=True))"""
grasping_error_str = """ObjectNotGraspedErrorModel(obj=Object(name='cup',concept='Cup', color='blue'),robot=Object(name='robot', concept='robot'), arm=Arms.LEFT, grasp=Grasp.TOP)"""
human_comment_str = "pick up the yellow cup not the blue cup"

data = {
    'action_designator': action_designator_str,
    'reason_for_failure': grasping_error_str,
    'human_comment': human_comment_str
}

response = requests.post(url, json=data)  # or data=data for form-data
print(response.json())