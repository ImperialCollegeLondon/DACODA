#ifndef CLASS_DRONE
#define CLASS_DRONE

#include <iostream>
#include <vector>
#include <chrono>
#include "omp.h"


using namespace std;

class Drone{
    private:

    int VectorLength;
    int DroneIndex;
    string DroneModel;
    int AircraftIndex;
    int TakeoffTime;
    int ArriveTime;

    double AircraftRadius;
    double DroneRadius;

    int longitude_col_no;
    int latitude_col_no;
    int heading_col_no;

    int PositionData_size;
    int TotalCols;
    vector<string> PositionData;
    string Airport;

    double* initial_long_pos;
    double* initial_lat_pos;
    double* initial_heading;

    int* longitude_factor_1st;
    int* latitude_factor_1st;

    void ColumnSelect_Index(int column_no, double* column_pointer, int DroneIndex);
    void SetInitialConditions();
    void FirstStage();
    void CubedVolume();
    void SecondStage();
    bool Collision();
    void EstimateCollisionPoint(int i);
    void Output(int run_no, string distance_from_airport);
    void Output_1File(int run_no, string distance_from_airport);
    void Deallocate();

    double* aircraft_longitude;
    double* aircraft_latitude;
    double* aircraft_altitude;
    double* aircraft_tracking;
    double* aircraft_groundspeed;
    double* aircraft_verticalRate;
    double* aircraft_FPA;

    double max_straight_speed;
    double max_ascend_speed;
    double max_descend_speed;
    double start_alt;

    int random_t_1st;

    double random_long;
    double random_lat;
    double random_alt;
    double heading_angle;

    double minCubeAlt;
    double maxCubeAlt;

    double gradientN;
    double gradientS;
    double gradientE;
    double gradientW;

    double interceptN;
    double interceptS;
    double interceptE;
    double interceptW;

    double* pointNW;
    double* pointNE;
    double* pointSE;
    double* pointSW;

    double boundaryN;
    double boundaryS;
    double boundaryE;
    double boundaryW;

    int depart_or_arrive;

    int* collision_index;

    double* longitude_vector;
    double* latitude_vector;
    double* altitude_vector;
    double* speed_vector;
    double* heading_vector;
    double* pitch_vector;
    double* collision_point;

    double* droneAreaLat;
    double* droneAreaLon;


    public:

    void CSVData(string FilePath);
    void SetInitialParameters(string Airport_input, string DroneModel_input, int Vector_length_input, int TotalCols_input, int drone_index_input, int aircraft_index_input, int takeoff_time_input, int arrive_time_input, double* air_long, double* air_lat, double* air_alt, double* air_track, double* air_speed, double* air_verticalRate, double aircraft_radius_input, double* droneAreaLat_input, double* droneAreaLon_input);
    void Simulation(int number_sims, double* total_collisions, double* local_collisions, string distance_from_airport, int run_number, double* total_sims);
    void Output_Collision_Num(double* local_collisions, string distance_from_airport);
    void ClearOutput(int Aircraft_Index, string distance_from_airport, string Airport_input, int depart_or_arrive);
    void ClearOutput_1File(string distance_from_airport, string Airport_input, int depart_or_arrive);
    void Output_1File_Collision_Num(double* total_collisions, string distance_from_airport, string Airport_input, int depart_or_arrive);
    void Average_ClearOutput_1File(string distance_from_airport, string Airport_input, string drone_model_input, int depart_or_arrive);
    void AverageOutputFile(string distance_from_airport, int run_number);
    void AverageOutputFile_LocalCollision(string Airport_input, string drone_model_input, double* local_collisions, string distance_from_airport, int run_number, int depart_or_arrive);
    void AverageOutputFile_TotalCollision(string Airport_input, string drone_model_input,  double* total_collisions, string distance_from_airport, double* total_sims, int max_run_number, int depart_or_arrive, chrono::seconds duration);

    
};

#endif
