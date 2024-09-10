
from pysimbotlib.core import PySimbotApp, Simbot, Robot, Util
from kivy.logger import Logger
from kivy.config import Config

import random
import csv
import os

# # Force the program to show user's log only for "info" level or more. The info log will be disabled.
# Config.set('kivy', 'log_level', 'debug')
Config.set('graphics', 'maxfps', 10)

class StupidRobot(Robot):

    RULE_LENGTH = 11
    NUM_RULES = 10

    def __init__(self, **kwarg):
        super(StupidRobot, self).__init__(**kwarg)
        self.RULES = [[0] * self.RULE_LENGTH for _ in range(self.NUM_RULES)]
        self.fitness = 0
        self.eat_count = 0
        self.collision_count = 0
        self.distance_to_food = 0

        # initial list of rules
        self.rules = [0.] * self.NUM_RULES
        self.turns = [0.] * self.NUM_RULES
        self.moves = [0.] * self.NUM_RULES

        self.fitness = 0

    def update(self):
        ''' Update method which will be called each frame
        '''        
        self.ir_values = self.distance()
        self.S0, self.S1, self.S2, self.S3, self.S4, self.S5, self.S6, self.S7 = self.ir_values
        self.target = self.smell()
        # Assign value to each rune variables ================================================#
        for i, RULE in enumerate(self.RULES):
            self.rules[i] = 1.0
            for k, RULE_VALUE in enumerate(RULE):
                if k < 8:
                    if RULE_VALUE % 5 == 1:
                        if k == 0: self.rules[i] *= self.S0_near()
                        elif k == 1: self.rules[i] *= self.S1_near()
                        elif k == 2: self.rules[i] *= self.S2_near()
                        elif k == 3: self.rules[i] *= self.S3_near()
                        elif k == 4: self.rules[i] *= self.S4_near()
                        elif k == 5: self.rules[i] *= self.S5_near()
                        elif k == 6: self.rules[i] *= self.S6_near()
                        elif k == 7: self.rules[i] *= self.S7_near()
                    elif RULE_VALUE % 5 == 2:
                        if k == 0: self.rules[i] *= self.S0_far()
                        elif k == 1: self.rules[i] *= self.S1_far()
                        elif k == 2: self.rules[i] *= self.S2_far()
                        elif k == 3: self.rules[i] *= self.S3_far()
                        elif k == 4: self.rules[i] *= self.S4_far()
                        elif k == 5: self.rules[i] *= self.S5_far()
                        elif k == 6: self.rules[i] *= self.S6_far()
                        elif k == 7: self.rules[i] *= self.S7_far()
                elif k == 8:
                    temp_val = RULE_VALUE % 6
                    if temp_val == 1: self.rules[i] *= self.smell_left()
                    elif temp_val == 2: self.rules[i] *= self.smell_center()
                    elif temp_val == 3: self.rules[i] *= self.smell_right()
                elif k==9: self.turns[i] = (RULE_VALUE % 181) - 90
                elif k==10: self.moves[i] = (RULE_VALUE % 21) - 10
        # ====================================================================================#

        # Finalize action ====================================================================#
        answerTurn = 0.0
        answerMove = 0.0
        for turn, move, rule in zip(self.turns, self.moves, self.rules):
            answerTurn += turn * rule
            answerMove += move * rule

        self.turn(answerTurn)
        self.move(answerMove)
        # ====================================================================================#


    def S0_near(self):
        if self.S0 <= 5: return 1.0
        elif self.S0 >= 30: return 0.0
        else: return 1 - (self.S0 / 25)

    def S0_far(self):
        if self.S0 <= 5: return 0.0
        elif self.S0 >= 30: return 1.0
        else: return self.S0 / 25
    
    def S1_near(self):
        if self.S1 <= 5: return 1.0
        elif self.S1 >= 30: return 0.0
        else: return 1 - (self.S1 / 25)
    
    def S1_far(self):
        if self.S1 <= 5: return 0.0
        elif self.S1 >= 30: return 1.0
        else: return self.S1 / 25
    
    def S2_near(self):
        if self.S2 <= 0: return 1.0
        elif self.S2 >= 30: return 0.0
        else: return 1 - (self.S2 / 30)
    
    def S2_far(self):
        if self.S2 <= 0: return 0.0
        elif self.S2 >= 30: return 1.0
        else: return self.S2 / 30
    
    def S3_near(self):
        if self.S3 <= 5: return 1.0
        elif self.S3 >= 30: return 0.0
        else: return 1 - (self.S3 / 25)
    
    def S3_far(self):
        if self.S3 <= 5: return 0.0
        elif self.S3 >= 30: return 1.0
        else: return self.S3 / 25
    
    def S4_near(self):
        if self.S4 <= 0: return 1.0
        elif self.S4 >= 100: return 0.0
        else: return 1 - (self.S4 / 100.0)
    
    def S4_far(self):
        if self.S4 <= 0: return 0.0
        elif self.S4 >= 100: return 1.0
        else: return self.S4 / 100.0
    
    def S5_near(self):
        if self.S5 <= 5: return 1.0
        elif self.S5 >= 30: return 0.0
        else: return 1 - (self.S5 / 25)
    
    def S5_far(self):
        if self.S5 <= 5: return 0.0
        elif self.S5 >= 30: return 1.0
        else: return self.S5 / 25
    
    def S6_near(self):
        if self.S6 <= 0: return 1.0
        elif self.S6 >= 30: return 0.0
        else: return 1 - (self.S6 / 30)
    
    def S6_far(self):
        if self.S6 <= 0: return 0.0
        elif self.S6 >= 30: return 1.0
        else: return self.S6 / 30
    
    def S7_near(self):
        if self.S7 <= 5: return 1.0
        elif self.S7 >= 30: return 0.0
        else: return 1 - (self.S7 / 25)
    
    def S7_far(self):
        if self.S7 <= 5: return 0.0
        elif self.S7 >= 30: return 1.0
        else: return self.S7 / 25
    
    def smell_right(self):
        if self.target >= 45: return 1.0
        elif self.target <= 0: return 0.0
        else: return self.target / 45.0
    
    def smell_left(self):
        if self.target <= -45: return 1.0
        elif self.target >= 0: return 0.0
        else: return 1-(-1*self.target)/45.0
    
    def smell_center(self):
        if self.target <= 45 and self.target >= 0: return self.target / 45.0
        if self.target <= -45 and self.target <= 0: return 1-(-1*self.target)/45.0
        else: return 0.0


    def calculate_fitness(self):
        # Normalize and weight different factors
        normalized_eat = self.eat_count / 2  # Assuming eat count is around 2
        normalized_collision = 1 - (self.collision_count / 100)  # Penalize collisions
        normalized_distance = 1 - (self.distance_to_food / 1000)  # Assuming max distance is 1000

        self.fitness = (
            0.5 * normalized_eat +
            0.3 * normalized_collision +
            0.2 * normalized_distance
        )

def write_rule(robot, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(robot.RULES)

def read_rule(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        return [list(map(int, row)) for row in reader]

# initializing next generation robot list
next_gen_robots = list()
# random 3 bot and pick the best one
def tournament_select(population, tournament_size=3):
    selected = random.sample(population, tournament_size)
    return max(selected, key=lambda robot: robot.fitness)
# combile the rule from parent 1 and 2 by random ratio
def crossover(parent1, parent2):
    child = StupidRobot()
    for i in range(StupidRobot.NUM_RULES):
        crossover_point = random.randint(1, StupidRobot.RULE_LENGTH - 1)
        child.RULES[i] = parent1.RULES[i][:crossover_point] + parent2.RULES[i][crossover_point:]
    return child
# probable 10% to random assign numbers to the rule (test each rule separately)
def mutate(robot, mutation_rate=0.01):
    for i in range(StupidRobot.NUM_RULES):
        for j in range(StupidRobot.RULE_LENGTH):
            if random.random() < mutation_rate:
                robot.RULES[i][j] = random.randrange(256)

def before_simulation(simbot: Simbot):
    if simbot.simulation_count == 0:
        Logger.info("GA: initial population")
        for robot in simbot.robots:
            for i in range(StupidRobot.NUM_RULES):
                for k in range(StupidRobot.RULE_LENGTH):
                    robot.RULES[i][k] = random.randrange(256)
    else:
        Logger.info("GA: using rules from previous generation")
        for simbot_robot, robot_from_last_gen in zip(simbot.robots, next_gen_robots):
            simbot_robot.RULES = robot_from_last_gen.RULES

def after_simulation(simbot: Simbot):
    Logger.info("GA: Start GA Process ...")

    # Calculate fitness for each robot
    for robot in simbot.robots:
        robot.calculate_fitness()

    # Sort robots by fitness
    sorted_robots = sorted(simbot.robots, key=lambda r: r.fitness, reverse=True)

    # Elitism: keep the best robot
    next_gen_robots.clear()
    next_gen_robots.append(sorted_robots[0])

    # Generate new population
    while len(next_gen_robots) < len(simbot.robots):
        parent1 = tournament_select(simbot.robots)
        parent2 = tournament_select(simbot.robots)
        child = crossover(parent1, parent2)
        mutate(child)
        next_gen_robots.append(child)

    # Log best fitness
    best_fitness = sorted_robots[0].fitness
    Logger.info(f"Generation {simbot.simulation_count}: Best Fitness = {best_fitness}")

    # Write the best rule to file
    write_rule(sorted_robots[0], f"PyGASimbot 2021/PyGASimbot/Rule_log/best_gen{simbot.simulation_count}.csv")

if __name__ == '__main__':
    app = PySimbotApp(
        robot_cls=StupidRobot, 
        num_robots=10,
        theme='default',
        simulation_forever=True,
        max_tick=2000,
        interval=1/100.0,
        food_move_after_eat=False,
        customfn_before_simulation=before_simulation, 
        customfn_after_simulation=after_simulation
    )
    app.run()