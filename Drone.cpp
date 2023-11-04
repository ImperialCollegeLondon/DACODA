#include "Drone.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <boost/filesystem.hpp>

using namespace std;


void Drone::CSVData(string FilePath){
    ifstream csv;

    csv.open(FilePath);

    if (csv.fail()) {
        cout << "ERROR - Failed to open " << FilePath << endl;
    }

    string line;
    string element;
    
    // Seperating line by line and then element by element 
    while(getline(csv, line)){
        stringstream new_line(line);
        while (getline(new_line, element, ',')){
        PositionData.push_back(element);
        }
    }
    csv.close();

    PositionData_size = PositionData.size();
}

void Drone::SetInitialParameters(string Airport_input, string Drone_model_input, int Vector_length_input, int TotalCols_input, int drone_index_input, int aircraft_index_input, int takeoff_time_input, int arrive_time_input, double* air_long, double* air_lat, double* air_alt, double* air_track, double* air_speed, double* air_verticalRate, double aircraft_radius_input, double* droneAreaLat_input, double* droneAreaLon_input){
    VectorLength = Vector_length_input;
    TotalCols = TotalCols_input;
    DroneIndex = drone_index_input;
    DroneModel = Drone_model_input;
    AircraftIndex = aircraft_index_input;
    TakeoffTime = takeoff_time_input;
    ArriveTime = arrive_time_input;
    AircraftRadius = aircraft_radius_input;
    Airport = Airport_input;

    // Column numbers
    longitude_col_no = 1;
    latitude_col_no = 2;
    heading_col_no = 3;

    // Initialise starting positions for drones
    initial_long_pos = new double[1];
    initial_lat_pos = new double[1];
    initial_heading = new double[1];

    // Initialise collision index between drone and aircraft
    collision_index = new int[1];

    // Obtain starting positions of drone from CSV file 
    *initial_long_pos = stod(PositionData[((DroneIndex+1)*TotalCols) + longitude_col_no]);
    *initial_lat_pos = stod(PositionData[((DroneIndex+1)*TotalCols) + latitude_col_no]);
    *initial_heading = stod(PositionData[((DroneIndex+1)*TotalCols) + heading_col_no]);

    aircraft_longitude = new double[VectorLength];
    aircraft_latitude = new double[VectorLength];
    aircraft_altitude = new double[VectorLength];
    aircraft_groundspeed = new double[VectorLength];
    aircraft_tracking = new double[VectorLength];
    aircraft_verticalRate = new double[VectorLength];
    aircraft_FPA = new double[VectorLength];
    

    droneAreaLat = new double[4];
    droneAreaLon = new double[4];


    for(int i = 0; i < VectorLength; ++i){
        aircraft_longitude[i] = air_long[i];
        aircraft_latitude[i] = air_lat[i];
        aircraft_altitude[i] = air_alt[i];
        aircraft_groundspeed[i] = air_speed[i];
        aircraft_tracking[i] = air_track[i];
        aircraft_verticalRate[i] = air_verticalRate[i];
        if(aircraft_groundspeed[i] <= 0){
            aircraft_FPA[i] = 0.0;    
        }
        else{
            aircraft_FPA[i] = (atan(aircraft_verticalRate[i] / aircraft_groundspeed[i]) * 180.0) / M_PI;
        }

        

    }
    
    for(int i = 0; i < 4; ++i){
        droneAreaLat[i] = droneAreaLat_input[i];
        droneAreaLon[i] = droneAreaLon_input[i];
    }

    if(aircraft_altitude[0] != 0){
        depart_or_arrive = 1; // ARRIVAL 
    }
    else{
        depart_or_arrive = 0; // DEPART
    }
    
    // THIS WILL BE ALTERED
    start_alt = 100.0; // Starting Altitude - m

    // POTENTIALL LOOK TO SHIFT THIS CHANGE
    if(DroneModel == "Mavic_3"){
        max_straight_speed = 19.0;  // m/s
        max_ascend_speed = 8.0;     // m/s
        max_descend_speed = 6.0;    // m/s
        DroneRadius = 0.19;         // m
    }

    if(DroneModel == "Mini_2"){
        max_straight_speed = 16.0;  // m/s
        max_ascend_speed = 5.0;     // m/s
        max_descend_speed = 3.5;    // m/s
        DroneRadius = 0.1065;       // m
    }

}


void Drone::ClearOutput(int Aircraft_Index, string distance_from_airport, string Airport_input, int depart_or_arrive){
    string aircraft_index = to_string(Aircraft_Index);
    ofstream outfile;
    if (depart_or_arrive){ // ARRIVE
        outfile.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/Arrival/Drone_coords_" + aircraft_index + ".csv", ofstream::out | ofstream::trunc);
        outfile.close();
    }
    else{ // DEPART
        outfile.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/Depart/Drone_coords_" + aircraft_index + ".csv", ofstream::out | ofstream::trunc);
        outfile.close();
    }

}


void Drone::ClearOutput_1File(string distance_from_airport, string Airport_input, int depart_or_arrive){
    boost::filesystem::path full_path(boost::filesystem::current_path());
    boost::filesystem::path dstFolder = Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km"; // Makes folder if there is no folder
    boost::filesystem::create_directory(dstFolder);
    if(depart_or_arrive){// ARRIVE
        boost::filesystem::path dstFolder = Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/Arrival"; // Makes folder if there is no folder
        boost::filesystem::create_directory(dstFolder);
        ofstream outfile;
        outfile.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/Arrival/All_Drone_Collisions.csv", ofstream::out | ofstream::trunc);
        outfile.close();        
    }
    else{ // DEPART
        boost::filesystem::path dstFolder = Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/Depart"; // Makes folder if there is no folder
        boost::filesystem::create_directory(dstFolder);
        ofstream outfile;
        outfile.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/Depart/All_Drone_Collisions.csv", ofstream::out | ofstream::trunc);
        outfile.close();
    }
}


void Drone::Average_ClearOutput_1File(string distance_from_airport, string Airport_input, string drone_model_input, int depart_or_arrive){
    boost::filesystem::path full_path(boost::filesystem::current_path());
    boost::filesystem::path dstFolder = Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km"; // Makes folder if there is no folder
    boost::filesystem::create_directory(dstFolder);

    // Make folder for Drone Model
    boost::filesystem::path ndstFolder = Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/" + drone_model_input; // Makes folder if there is no folder
    boost::filesystem::create_directory(ndstFolder);

    ofstream outfile;

    if(depart_or_arrive){// ARRIVE
        boost::filesystem::path dstFolder = Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/" + drone_model_input + "/Arrival"; // Makes folder if there is no folder
        boost::filesystem::create_directory(dstFolder);
        outfile.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/" + drone_model_input + "/Arrival/collisionsArrive.csv", ofstream::out | ofstream::trunc);
    }

    else{ // DEPART
        boost::filesystem::path dstFolder = Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/" + drone_model_input + "/Depart"; // Makes folder if there is no folder
        boost::filesystem::create_directory(dstFolder);
        outfile.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/" + drone_model_input + "/Depart/collisionsDepart.csv", ofstream::out | ofstream::trunc);
    }

    outfile << "run_number,drone_longitude,drone_latitude,drone_altitude,drone_speed,drone_heading,drone_FPA,drone_index,aircraft_longitude,aircraft_latitude,aircraft_altitude,aircraft_speed,aircraft_heading,aircraft_FPA,aircraft_indextime" << '\n';
    outfile.close();
}


void Drone::SetInitialConditions(){
    // Allocate memory to vectors
    longitude_vector = new double[VectorLength];
    latitude_vector = new double[VectorLength];
    altitude_vector = new double[VectorLength];
    speed_vector = new double[VectorLength];
    heading_vector = new double[VectorLength];
    pitch_vector = new double[VectorLength];

    pointNW = new double[2];
    pointNE = new double[2];
    pointSE = new double[2];
    pointSW = new double[2];

    collision_point = new double[3];

    // Initialise vectors
    longitude_vector[0] = *initial_long_pos; // Initialising longitude
    latitude_vector[0] = *initial_lat_pos; // Initialising latitude
    heading_vector[0] = *initial_heading; // Initialising heading
    altitude_vector[0] = start_alt; // 100m starting altitude
    speed_vector[0] = max_straight_speed; // 19m/s max strightline speed
    pitch_vector[0] = 0.0;
}


void Drone::FirstStage(){
    int min_t_1st = 1;
    int max_t_1st;

    if(depart_or_arrive){ // ARRIVE
        max_t_1st = ArriveTime;
    }
    else{ // DEPART
        max_t_1st = TakeoffTime;
    }
    

    //random_device seed;d
    //mt19937 engine(seed());
    default_random_engine engine{random_device{}()};
    uniform_int_distribution<int> dist(min_t_1st, max_t_1st); // uniform, unbiased

    random_t_1st = dist(engine);
    //random_t_1st = rand()%(max_t_1st-min_t_1st + 1) + min_t_1st;


    for(int i = 1; i < random_t_1st; ++i){
        longitude_vector[i] = longitude_vector[i-1] + sin(heading_vector[i-1])*max_straight_speed;
        latitude_vector[i] = latitude_vector[i-1] + cos(heading_vector[i-1])*max_straight_speed;
        heading_vector[i] = heading_vector[i-1];
        speed_vector[i] = max_straight_speed;
        altitude_vector[i] = start_alt;
        pitch_vector[i] = 0.0;
    }
}

void Drone::CubedVolume(){
    /*
    Need to make a loop which can use the runwayName from the Aircraft class to identify the required runway drone location needed.
    For departure flights it would need to be flipped, maybe have the change in the csv file from Python and label the heading 'drone_target_runway'
    */       

    minCubeAlt = 0;
    maxCubeAlt = 500;

    // 0-NW 1-NE 2-SE 3-SW  -- Points for droneAreaLat and droneAreaLon
    
    pointNW[0] = droneAreaLat[0];
    pointNE[0] = droneAreaLat[1];
    pointSE[0] = droneAreaLat[2];
    pointSW[0] = droneAreaLat[3];

    pointNW[1] = droneAreaLon[0];
    pointNE[1] = droneAreaLon[1];
    pointSE[1] = droneAreaLon[2];
    pointSW[1] = droneAreaLon[3];
    
    
    gradientN = (pointNE[0] - pointNW[0])/(pointNE[1] - pointNW[1]);
    gradientS = (pointSE[0] - pointSW[0])/(pointSE[1] - pointSW[1]);
    gradientE = (pointNE[0] - pointSE[0])/(pointNE[1] - pointSE[1]);
    gradientW = (pointNW[0] - pointSW[0])/(pointNW[1] - pointSW[1]);

    interceptN = (pointNE[0] - gradientN*pointNE[1]);
    interceptS = (pointSE[0] - gradientS*pointSE[1]);
    interceptE = (pointNE[0] - gradientE*pointNE[1]);
    interceptW = (pointNW[0] - gradientW*pointNW[1]);

    
    
}


void Drone::SecondStage(){
    CubedVolume();
    
    // Uniform distribution - LONGITUDE
    uniform_real_distribution<double> long_dist(min(pointNW[1],pointSW[1]), max(pointNE[1],pointSE[1])); // uniform, unbiased

    // Uniform distribution - LATITUDE
    uniform_real_distribution<double> lat_dist(min(pointSE[0],pointSW[0]), max(pointNE[0],pointNW[0])); // uniform, unbiased

    // Uniform distribution - ALTITUDE
    uniform_real_distribution<double> alt_dist(minCubeAlt, maxCubeAlt); // uniform, unbiased

    bool validPosition = false;
    while (!validPosition)
    {
        default_random_engine engine_1{random_device{}()};
        default_random_engine engine_2{random_device{}()};
        default_random_engine engine_3{random_device{}()};

        random_long = long_dist(engine_1);
        random_lat = lat_dist(engine_2);
        random_alt = alt_dist(engine_3);

        boundaryN = random_long*gradientN + interceptN;
        boundaryS = random_long*gradientS + interceptS; 
        boundaryE = (random_lat - interceptE)/gradientE;
        boundaryW = (random_lat - interceptW)/gradientW;
        

        if(random_lat <= boundaryN && random_lat >= boundaryS && random_long >= boundaryW && random_long <= boundaryE){
            validPosition = true;
        }
        
    }
        

    double pitch_angle;
    double modulus_long_lat;
    double velocity_factor;

    int last_index = random_t_1st - 1;

    double long_diff = abs(random_long - longitude_vector[last_index]);
    double lat_diff = abs(random_lat - latitude_vector[last_index]);
    double alt_diff = random_alt - altitude_vector[last_index];

    // TOP LEFT
    if(random_long <= longitude_vector[last_index] && random_lat >= latitude_vector[last_index]){
        heading_angle = 2*M_PI - atan(long_diff / lat_diff);
    }
    // BOTTOM LEFT
    else if(random_long <= longitude_vector[last_index] && random_lat <= latitude_vector[last_index]){
        heading_angle = 1.5*M_PI - atan(lat_diff / long_diff);
    }
    // BOTTOM RIGHT
    else if(random_long >= longitude_vector[last_index] && random_lat <= latitude_vector[last_index]){
        heading_angle = M_PI - atan(long_diff / lat_diff);
    }
    // TOP RIGHT
    else if(random_long >= longitude_vector[last_index] && random_lat >= latitude_vector[last_index]){
        heading_angle = atan(long_diff / lat_diff);
    }
    
    modulus_long_lat = sqrt(long_diff*long_diff + lat_diff*lat_diff);

    pitch_angle = atan(alt_diff / modulus_long_lat);

    if(alt_diff < 0){
        velocity_factor = (max_straight_speed - (2*(max_straight_speed - max_descend_speed)/M_PI)*pitch_angle);
    }
    else{
        velocity_factor = (max_straight_speed - (2*(max_straight_speed - max_ascend_speed)/M_PI)*pitch_angle);
    }
    



    for(int i = random_t_1st; i < VectorLength; ++i){
        longitude_vector[i] = longitude_vector[i-1] + sin(heading_angle) * velocity_factor;
        latitude_vector[i] = latitude_vector[i-1] + cos(heading_angle) * velocity_factor;
        altitude_vector[i] = altitude_vector[i-1] + sin(pitch_angle) * velocity_factor;
        speed_vector[i] = velocity_factor;
        heading_vector[i] = heading_angle;
        pitch_vector[i] = pitch_angle;

        // NEW - Make sure that drone does not fly below DroneRadius + 1 altitude
        if(altitude_vector[i] < DroneRadius + 1){
            altitude_vector[i] = DroneRadius + 1;
        }
    }
}

void Drone::Output_1File(int run_no, string distance_from_airport){
    string aircraft_index = to_string(AircraftIndex);
    ofstream outfile1;
    if(depart_or_arrive){ // ARRIVE
        outfile1.open(Airport + "/Drone_Collisions_" + distance_from_airport + "_km/Arrival/All_Drone_Collisions.csv", ofstream::out | ofstream::app);
    }
    else{ // DEPART
        outfile1.open(Airport + "/Drone_Collisions_" + distance_from_airport + "_km/Depart/All_Drone_Collisions.csv", ofstream::out | ofstream::app);
    }

    outfile1.precision(10);
    for (int i=0; i < VectorLength; ++i){
        if(i == *collision_index){
            outfile1 << longitude_vector[i] << "," << latitude_vector[i] << "," << altitude_vector[i] << "," << run_no << "," << AircraftIndex << "," << 1 << '\n';    
        }
        else{
            outfile1 << longitude_vector[i] << "," << latitude_vector[i] << "," << altitude_vector[i] << "," << run_no << "," << AircraftIndex << "," << 0 << '\n';
        }
    }
    outfile1 << '\n';
    outfile1.close();
}

void Drone::Output(int run_no, string distance_from_airport){
    string aircraft_index = to_string(AircraftIndex);
    ofstream outfile;
    if(depart_or_arrive){ // ARRIVE
        outfile.open(Airport + "/Drone_Collisions_" + distance_from_airport + "_km/Arrival/Drone_coords_" + aircraft_index + ".csv", ofstream::out | ofstream::app);
    }
    else{ // DEPART
        outfile.open(Airport + "/Drone_Collisions_" + distance_from_airport + "_km/Depart/Drone_coords_" + aircraft_index + ".csv", ofstream::out | ofstream::app);
    }

    outfile.precision(10);
    for (int i=0; i < VectorLength; ++i){
        if(i == *collision_index){
            outfile << longitude_vector[i] << "," << latitude_vector[i] << "," << altitude_vector[i] << "," << run_no << "," << 1 << '\n';    
        }
        else{
            outfile << longitude_vector[i] << "," << latitude_vector[i] << "," << altitude_vector[i] << "," << run_no << "," << 0 << '\n';
        }
    }
    outfile << '\n';
    outfile.close();
}

void Drone::Output_Collision_Num(double* local_collisions, string distance_from_airport){
    string aircraft_index = to_string(AircraftIndex);
    ofstream outfile;
    if(depart_or_arrive){ // ARRIVE
        outfile.open(Airport + "/Drone_Collisions_" + distance_from_airport + "_km/Arrival/Drone_coords_" + aircraft_index + ".csv", ofstream::out | ofstream::app);
    }
    else{ // DEPART
        outfile.open(Airport + "/Drone_Collisions_" + distance_from_airport + "_km/Depart/Drone_coords_" + aircraft_index + ".csv", ofstream::out | ofstream::app);
    }
    
    outfile << *local_collisions;
    outfile.close();
}

void Drone::Output_1File_Collision_Num(double* total_collisions, string distance_from_airport, string Airport_input, int depart_or_arrive){
    ofstream outfile1;
    if(depart_or_arrive){ // ARRIVE
        outfile1.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/Arrival/All_Drone_Collisions.csv", ofstream::out | ofstream::app);
    }
    else{ // DEPART
        outfile1.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/Depart/All_Drone_Collisions.csv", ofstream::out | ofstream::app);
    }
    outfile1 << *total_collisions;
    outfile1.close();
}

void Drone::AverageOutputFile(string distance_from_airport, int run_number){
    ofstream outfile;
    if(depart_or_arrive){ // ARRIVE
        outfile.open(Airport + "/Drone_Collisions_" + distance_from_airport + "_km/" + DroneModel + "/Arrival/collisionsArrive.csv", ofstream::out | ofstream::app);
    }
    else{ // DEPART
        outfile.open(Airport + "/Drone_Collisions_" + distance_from_airport + "_km/" + DroneModel + "/Depart/collisionsDepart.csv", ofstream::out | ofstream::app);
    }

    outfile.precision(10);
    outfile << run_number << "," << longitude_vector[*collision_index] << "," << latitude_vector[*collision_index] << "," << altitude_vector[*collision_index] << "," << speed_vector[*collision_index] << "," << heading_vector[*collision_index]*180/M_PI << "," << pitch_vector[*collision_index]*180/M_PI << "," << DroneIndex << "," << aircraft_longitude[*collision_index] << "," << aircraft_latitude[*collision_index] << "," << aircraft_altitude[*collision_index] << "," << aircraft_groundspeed[*collision_index] << "," << aircraft_tracking[*collision_index] << "," << aircraft_FPA[*collision_index] << "," << AircraftIndex << "," << *collision_index << '\n';    
    outfile.close();
}

void Drone::AverageOutputFile_LocalCollision(string Airport_input, string drone_model_input, double* local_collisions, string distance_from_airport, int run_number, int depart_or_arrive){
    ofstream outfile;
    if(depart_or_arrive){ // ARRIVE
        outfile.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/" + drone_model_input + "/Arrival/collisionsArrive.csv", ofstream::out | ofstream::app);
    }
    else{ // DEPART
        outfile.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/" + drone_model_input + "/Depart/collisionsDepart.csv", ofstream::out | ofstream::app);
    }
    
    outfile << run_number << ",Local Collision Number," << *local_collisions << '\n';
    outfile << '\n';
    outfile.close();
}

void Drone::AverageOutputFile_TotalCollision(string Airport_input, string drone_model_input, double* total_collisions, string distance_from_airport, double* total_sims, int max_run_number, int depart_or_arrive, chrono::seconds duration){
    ofstream outfile;
    if(depart_or_arrive){ // ARRIVE
        outfile.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/" + drone_model_input + "/Arrival/collisionsArrive.csv", ofstream::out | ofstream::app);
    }
    else{ // DEPART
        outfile.open(Airport_input + "/Drone_Collisions_" + distance_from_airport + "_km/" + drone_model_input + "/Depart/collisionsDepart.csv", ofstream::out | ofstream::app);
    }
    
    outfile << "Total time duration in seconds," << duration.count() << '\n';
    outfile << "Total Collision Number," << *total_collisions << '\n';
    outfile << "Total simulation runs," << *total_sims << '\n';
    outfile << "Total Monte-Carlo runs," << max_run_number << '\n';
    outfile.precision(10);
    outfile << "Average number of Collisions per Monte-Carlo run," << *total_collisions/(max_run_number*1.0) << '\n';
    outfile << "Average percentage of collision," << (*total_collisions / *total_sims)*100.0; 
    outfile.close();
}

void Drone::Deallocate(){
    delete[] longitude_vector;
    delete[] latitude_vector;
    delete[] altitude_vector;
    delete[] speed_vector;
    delete[] heading_vector;
    delete[] pitch_vector;
    delete[] collision_index;
    delete[] pointNE;
    delete[] pointNW;
    delete[] pointSE;
    delete[] pointSW;
    delete[] collision_point;
}

bool Drone::Collision(){
    for(int i = 0; i < VectorLength; ++i){
        double distance = sqrt(
            (longitude_vector[i] - aircraft_longitude[i]) * (longitude_vector[i] - aircraft_longitude[i]) +
            (latitude_vector[i] - aircraft_latitude[i]) * (latitude_vector[i] - aircraft_latitude[i]) + 
            (altitude_vector[i] - aircraft_altitude[i]) * (altitude_vector[i] - aircraft_altitude[i])
        );
        if (distance <= (DroneRadius + AircraftRadius)){
            *collision_index = i;
            //EstimateCollisionPoint(i);
            return 1;
        }
    }
    return 0;
}



void Drone::Simulation(int number_sims, double* total_collisions, double* local_collisions, string distance_from_airport, int run_number, double* total_sims){
    SetInitialConditions();
    
    for(int i = 0; i < number_sims; ++i){
        *total_sims += 1;
        FirstStage();   // Drone heads towards centre of runway
        SecondStage();  // Drone heads towards random coords in volume
        if (Collision()){
            AverageOutputFile(distance_from_airport, run_number);
            *local_collisions += 1;
            *total_collisions += 1;
        }
    }
    Deallocate();

}

void Drone::EstimateCollisionPoint(int i){
    // Direction Vector
    double direction_x = -sin(heading_vector[i]) * cos(pitch_vector[i]);
    double direction_y = cos(heading_vector[i]) * cos(pitch_vector[i]);
    double direction_z = sin(pitch_vector[i]);

    // Normalise Direction Vector
    double magnitude = sqrt(direction_x * direction_x + direction_y * direction_y + direction_z * direction_z);
    double normal_direction_x = direction_x / magnitude;
    double normal_direction_y = direction_y / magnitude;
    double normal_direction_z = direction_z / magnitude;

    // Displacement Vector
    double displacement_x = normal_direction_x * (DroneRadius + AircraftRadius);
    double displacement_y = normal_direction_y * (DroneRadius + AircraftRadius);
    double displacement_z = normal_direction_z * (DroneRadius + AircraftRadius);

    // Magnititude of displacement vector
    double magnitude_disp = sqrt(displacement_x * displacement_x + displacement_y * displacement_y + displacement_z * displacement_z);
    
    // Coliision point on the surface of aircraft sphere
    collision_point[0] = (displacement_x/magnitude_disp) * AircraftRadius;
    collision_point[1] = (displacement_y/magnitude_disp) * AircraftRadius;
    collision_point[2] = (displacement_z/magnitude_disp) * AircraftRadius;

}