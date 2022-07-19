# CS 471 HW5
## Thomas Bowidowicz, Mike Adams, F21

- The homework Latex report labeled homework5.pdf is found in report subdirectory of the hw5 directory.
- There are three subdirectories in the hw5 directory:
    1. code - This subdirectory contains all of the source code used in the project and is where the results can be recreated
    2. files - This subdriectory contains all of the output from the code including the plots and the .mp4 movie files.
    3. report - This subdirectory contains the final Latex report

- To recreate the tables and plots found in the report, follow the following instructions for each question:

Tasks 2/3:
- To recreate the results of the task 1 of the homework:
    1. Cd into the code directory: hw5/code/
    2. Tune the parameters to the desired settings. The impact of each parameter is detailed in the full report, but here is a quick guide:
        - N = population of birds
        - gamma_1 = the tastiness of food or how strongly the leader will go towards the food
        - gamma_2 = the charisma of the leader or how intensely the flock will follow the leader
        - alpha = the degree of the arc of the movement of the food source
            - This also depends on the food flag which can cause the food to move or not. 1 = circularized movement, 0 = stationary food at origin
        - kappa = the intensity of the flocking force or how much the birds will be pulled together
        - rho = the intensity of the repelling force or how much the birds will be pushed apart
        - delta = the intensity with which the birds are pushed apart
        - Note: The default parameters are: N = 30, gamma_1 = 2.0, gamma_2 = 8.0, alpha = 0.4, kappa = 4.0, rho = 2.0, delta = 0.5
    3. Run the following command with the the hw5_skeleton.py file: python hw5_skeleton.py
    4. The code will generate both a movie .mp4 file and a diameter_plot .png plot. 
    5. The file name for both of the movie and the diagram follows the following format:
    movieordiameter_plot_N_gamma_1_gamma_2_alpha_kappa_rho_delta_.filetype
