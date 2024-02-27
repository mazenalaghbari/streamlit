# streamlit

Below is a sample code showcasing Streamlit functionality to enable data science projects for end-users effortlessly. The application includes connectivity with a MySQL database, allowing users to navigate through index pages. Additionally, it features a machine-learning model that permits users to input data and receive corresponding outputs from the model.


# Creating the 'users' table with columns for user information
CREATE TABLE `users` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `name` VARCHAR(255) NOT NULL,
    `email` VARCHAR(255) NOT NULL,
    `password` VARCHAR(255) NOT NULL
);
