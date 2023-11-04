#include "Aircraft.h"
#include "Drone.h"
#include <iostream>
#include <omp.h>
#include <boost/program_options.hpp>
#include <iomanip>
#include <sstream>
#include <chrono>

namespace po = boost::program_options;
using namespace std;

int main(int argc, char* argv[]){

  po::options_description opts("Available options.");
  opts.add_options()
    ("airport", po::value<string>()->default_value("LHR"), "Airport Code.")
    ("date", po::value<string>()->default_value("2017-06-30"), "Date of simulation.")
    ("drone_model", po::value<string>()->default_value("Mavic_3"), "Drone Model: 'Mavic_3' or 'Mini_2'.")
    ("depart_arrive", po::value<int>()->default_value(1), "Departure (0) or Arrival (1). ")
    ("distance", po::value<double>()->default_value(1.0), "Inital distance of drones from centre of airport.")
    ("total_runs", po::value<int>()->default_value(1), "Total run number of program.")
    ("help", "Print help message.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);

  if(vm.count("help")){
    cout << opts << endl;
    return 1;
  }
  
  stringstream stream;
  stream << fixed << setprecision(1) << vm["distance"].as<double>();

  string Airport = vm["airport"].as<string>();
  string Date = vm["date"].as<string>();
  string DroneModel = vm["drone_model"].as<string>();
  int depart_or_arrive = vm["depart_arrive"].as<int>();
  string distance_from_airport = stream.str();
  int max_run_number = vm["total_runs"].as<int>();

  string FilePath_aircraft;

  if (!depart_or_arrive){ // DEPART
    FilePath_aircraft = Airport + "/TAKEOFF_UTM_DEPART_A320 " + Airport + " time - " + Date + " 04-00 to " + Date + " 23-45.csv";
  }
  else{ // ARRIVE
    FilePath_aircraft = Airport + "/LANDING_UTM_ARRIVE_A320 " + Airport + " time - " + Date + " 04-00 to " + Date + " 23-45.csv";
  } 
  

  int no_col_aircraft = 24; // Total number of  Columns in Aircraft CSV file
  double aircraft_radius = 6.55; // Frontal Area of aircraft (A320ceo) - distance from engine edge to centre line 

  string FilePath_drone = Airport + "/DroneStartPos/drone_inital_positions_" + distance_from_airport + "km.csv"; /// THIS CHANGES EVERYTHING
  string droneAreaFilePath = Airport + "/" + Airport + "_drone_area.csv";
  int no_col_drone = 4; // Total number of columns in Drone CSV file
  int drone_total = 99; // Total number of drones around airport
  int droneAreaRows = 6; // Total number of rows in DroneArea csv file


  double* total_collisions = new double[1];
  *total_collisions = 0;

  double* total_sims = new double[1];
  *total_sims = 0;

  Aircraft Aircraft;
  Drone Drone;

  Aircraft.Set_Parameters_and_Data(FilePath_aircraft, droneAreaFilePath, no_col_aircraft);
  Drone.Average_ClearOutput_1File(distance_from_airport, Airport, DroneModel, depart_or_arrive);
  Drone.CSVData(FilePath_drone);

  
  // Start Timer
  chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

  
  for(int run_number=0; run_number < max_run_number; ++run_number){
    int i;
    double* local_collisions = new double[1];
    *local_collisions = 0;

    #pragma omp parallel for private(i) firstprivate(Aircraft,Drone)
    for(i=0; i < Aircraft.Aircraft_Index_size-1; ++i){
      Aircraft.Vector_Allocation(i, droneAreaRows);
      
      for(int j=0; j < drone_total; ++j){
        
        Drone.SetInitialParameters(Airport, DroneModel, Aircraft.Vector_length, no_col_drone, j, i, Aircraft.takeoff_t, Aircraft.arrive_t, Aircraft.longitude_vector, Aircraft.latitude_vector, Aircraft.altitude_vector, Aircraft.track_vector, Aircraft.groundspeed_vector, Aircraft.verticalRate_vector, aircraft_radius, Aircraft.droneAreaLat, Aircraft.droneAreaLon);
        Drone.Simulation(10000, total_collisions, local_collisions, distance_from_airport, run_number, total_sims);
        
      }
      Aircraft.Deallocation();
    }
    Drone.AverageOutputFile_LocalCollision(Airport, DroneModel, local_collisions, distance_from_airport, run_number, depart_or_arrive);
    cout << "Number of collisions: " << *local_collisions << endl;
  }
  
  // End Timer
  chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();

  chrono::seconds duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
  Drone.AverageOutputFile_TotalCollision(Airport, DroneModel, total_collisions, distance_from_airport, total_sims, max_run_number, depart_or_arrive, duration);
 }
