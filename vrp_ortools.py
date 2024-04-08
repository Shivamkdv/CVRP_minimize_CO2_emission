import time 
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    #print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    total_distance1=0#Changes
    ans=[]#Changes
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        route_distance1=0#Changes
        vehicle_route=[]#Changes
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            vehicle_route.append(node_index)#Changes
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            node_index1=manager.IndexToNode(index)#Changes
            route_distance1+=data['distance_matrix'][node_index][node_index1]#Changes
            #print(routing.GetArcCostForVehicle(previous_index, index, vehicle_id))
        vehicle_route.append(manager.IndexToNode(index))
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Distance of the route (my code) :{}m\n'.format(route_distance1)#Changes
        plan_output += 'Load of the route: {}\n'.format(route_load)
        #print(plan_output)
        total_distance += route_distance
        total_distance1+=route_distance1
        total_load += route_load
        ans.append(vehicle_route)#Changes
    #print('Total distance of all routes: {}m'.format(total_distance))
    #print('Total distance of all routes (my code):{}m\n'.format(total_distance1))#Changes
    #print('Total load of all routes: {}'.format(total_load))
    return ans,total_distance1

def distance_matrix(locations,multiplier=1):
  distance=[]
  for x1,y1 in zip(locations[0],locations[1]):
    dist=[]
    for x2,y2 in zip(locations[0],locations[1]):
      if (x1,y1)==(x2,y2):
        dist.append(0)
      else:
        dist.append(math.hypot((x1-x2),(y1-y2))*multiplier)
    distance.append(dist)

  return distance

def create_data_model1(locations,demand,multiplier,capacity):
    """ Stores the data for the problem.
        Locations : 2 * (Number of Customer+1)
        demand : torch.Size([Number of Customer +1])
        multiplier : is 1000 (mostly)
        Capacity : Max Capacity of Vehicle 
        #Assuming Max demand of any customer is 9
        For stafey Reason we inceares the number of vehicle by 3
    """
    # print(f'locations: {locations} \n Multiplier: {multiplier}')

    data = {}
    n=len(locations[0])-1
    number_vehicle=20
    #number_vehicle=10000
    #print("Number of Customer : {} Capactiy :{} Number of vehicle : {}".format(n,capacity,number_vehicle))
    data['distance_matrix'] = distance_matrix(locations,multiplier)
    #print(f'distance_matrix: {data['distance_matrix']}')
    data['demands'] = demand
    data['vehicle_capacities'] = [capacity]*number_vehicle
    data['num_vehicles'] = number_vehicle
    data['depot'] = 0
    return data

def solve_vrp(location,demand,multiplier,capacity):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model1(location,demand,multiplier,capacity)
    #print(f'Data: {data}')

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    #print(f'Manager: {manager.GetNumberOfNodes}')

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # current_capacity_1 = routing.GetArcCostForVehicle(0, 1, 0)  # Assuming vehicle 0, adjust if needed.
    # print(f'current_capacity_1: {current_capacity_1}')

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
        # from_node = manager.IndexToNode(from_index)
        # to_node = manager.IndexToNode(to_index)
        
        # # Calculate the distance between nodes.
        # distance = data['distance_matrix'][from_node][to_node]
        
        # # Get the current capacity of the vehicle.
        # current_capacity = routing.GetArcCostForVehicle(from_index, to_index, 0)  # Assuming vehicle 0, adjust if needed.
        # print(f'Current capacity: {current_capacity}')
        # # Modify the cost calculation based on your requirement.
        # custom_cost = distance * current_capacity
        
        # return custom_cost

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    #print(f'transit_callback_index 1: {transit_callback_index}')
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    #print(f'transit_callback_index 2: {transit_callback_index}')
    

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    #It set time it will run the solutions
    #search_parameters.time_limit.FromSeconds(5)
    
    #It set the number of solutions
    search_parameters.solution_limit = 650

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        route,distance=print_solution(data, manager, routing, solution)
        #print(f"Number of solutions :{solution.solutions}")
        #print(route)
        #solve(route)
        #print(distance/multiplier)
        #print(data['distance_matrix'])
        return (distance/multiplier),route

def solve_vrp_time(location,demand,multiplier,capacity,time):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model1(location,demand,multiplier,capacity)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
        # from_node = manager.IndexToNode(from_index)
        # to_node = manager.IndexToNode(to_index)
        
        # # Calculate the distance between nodes.
        # distance = data['distance_matrix'][from_node][to_node]
        
        # # Get the current capacity of the vehicle.
        # current_capacity = routing.GetArcCostForVehicle(from_index, to_index, 0)  # Assuming vehicle 0, adjust if needed.
        # print(f'Current capacity: {current_capacity}')
        # # Modify the cost calculation based on your requirement.
        # custom_cost = distance * current_capacity
        
        # return custom_cost

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    #It set time it will run the solutions
    search_parameters.time_limit.FromSeconds(time)
    
    #It set the number of solutions
    #search_parameters.solution_limit = 650

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        route,distance=print_solution(data, manager, routing, solution)
        #print(f"Number of solutions :{solution.solutions}")
        #print(route)
        #solve(route)
        #print(distance/multiplier)
        #print(data['distance_matrix'])
        return (distance/multiplier),route

def solve_ortools(locations,demands,capacity,multiplier=1):
    # print('solve ortools')
    """ 
    This function return vehicle routes of Capacity Vehicle Routing Problem.
    Parameters:
    locations   : location of customer plus depot and depot must be at zero index
                 shape : (2 * Num_Node)
    demands     : demand for each customer plus depot
                 shape : (Num_Node)
    capacity   : Capacity of vehicle (Since in our case we consider only homogenous vehicle, 
                 capacity is integer)
    Multiplier : If input coordinate are in between (0,1) than multiplier is 1e4
                 and if input coordinates are larger than 1 than multiplier is 1 can be used
    Return Type: Tuple (routes,distance travelled) and routes contain routes of all vehicle
    
    NOTE : demands are considered between (0,1) so take it considered
    """
    # print(f'demands: {demands}')
    # print(f'capacity: {capacity}')
    
    demands=(demands*capacity).type(torch.int64)
    # print(f'demands after multiplication: {demands}')
    return solve_vrp(locations, demands, multiplier, capacity)

def solve_ortools_time(locations,demands,capacity,time,multiplier=1,):
    """ 
    This function return vehicle routes of Capacity Vehicle Routing Problem.
    Parameters:
    locations   : location of customer plus depot and depot must be at zero index
                 shape : (2 * Num_Node)
    demands     : demand for each customer plus depot
                 shape : (Num_Node)
    capacity   : Capacity of vehicle (Since in our case we consider only homogenous vehicle, 
                 capacity is integer)
    Multiplier : If input coordinate are in between (0,1) than multiplier is 1e4
                 and if input coordinates are larger than 1 than multiplier is 1 can be used
    Return Type: Tuple (routes,distance travelled) and routes contain routes of all vehicle
    
    NOTE : demands are considered between (0,1) so take it considered
    """
    demands=(demands*capacity).type(torch.int64)
    return solve_vrp_time(locations, demands, multiplier, capacity,time)


