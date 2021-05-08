CREATE DATABASE  IF NOT EXISTS `data_base_os` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci */;
USE `data_base_os`;
-- MySQL dump 10.13  Distrib 8.0.13, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: data_base_amdocs
-- ------------------------------------------------------
-- Server version	8.0.13

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
 SET NAMES utf8 ;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `all_changes`
--

DROP TABLE IF EXISTS `all_changes_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `all_changes_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `author` varchar(200) DEFAULT NULL,
  `created` datetime DEFAULT NULL,
  `from_string` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `to_string` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `field` varchar(200) DEFAULT NULL,
  `if_change_first_hour` int(11) DEFAULT NULL,
  `different_time_from_creat` float DEFAULT NULL,
  `is_first_setup` int(11) DEFAULT NULL,
  `chronological_number` int(11) NOT NULL,
  PRIMARY KEY (`issue_key`,`chronological_number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;



DROP TABLE IF EXISTS `changes_description_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `changes_description_os` (
  `issue_key` char(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `project_key` varchar(45) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `author` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `created` datetime DEFAULT NULL,
  `from_string` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  `to_string` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  `if_change_first_hour` int(11) DEFAULT NULL,
  `different_time_from_creat` float DEFAULT NULL,
  `is_first_setup` int(11) DEFAULT NULL,
  `is_diff_more_than_ten` int(11) DEFAULT NULL,
  `chronological_number` int(11) NOT NULL,
  # different from from str to to srt (the next after him)
  `ratio_different_char_next` float DEFAULT NULL,
  `ratio_different_word_next` float DEFAULT NULL,
  `num_different_char_minus_next` INT DEFAULT NULL,
  `num_different_char_plus_next` INT DEFAULT NULL,
  `num_different_char_all_next` INT DEFAULT NULL,
  `num_different_word_minus_next` INT DEFAULT NULL,
  `num_different_word_plus_next` INT DEFAULT NULL,
  `num_different_word_all_next` INT DEFAULT NULL,
  # different from from str to current(last) value (the final value)
  `ratio_different_char_last` float DEFAULT NULL,
  `ratio_different_word_last` float DEFAULT NULL,
  `num_different_char_minus_last` INT DEFAULT NULL,
  `num_different_char_plus_last` INT DEFAULT NULL,
  `num_different_char_all_last` INT DEFAULT NULL,
  `num_different_word_minus_last` INT DEFAULT NULL,
  `num_different_word_plus_last` INT DEFAULT NULL,
  `num_different_word_all_last` INT DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`chronological_number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `changes_description`
--

--
-- Table structure for table `changes_sprint`
--

DROP TABLE IF EXISTS `changes_sprint_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `changes_sprint_os` (
  `issue_key` char(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `project_key` varchar(45) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `author` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `created` datetime DEFAULT NULL,
  `from_string` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `to_string` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `if_change_first_hour` int(11) DEFAULT NULL,
  `different_time_from_creat` float DEFAULT NULL,
  `is_first_setup` int(11) DEFAULT NULL,
  `chronological_number` int(11) NOT NULL,
  PRIMARY KEY (`issue_key`,`chronological_number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `changes_sprint`
--


--
-- Table structure for table `changes_story_points`
--

DROP TABLE IF EXISTS `changes_story_points_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `changes_story_points_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `author` varchar(200) DEFAULT NULL,
  `created` datetime DEFAULT NULL,
  `from_string` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `to_string` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `if_change_first_hour` int(11) DEFAULT NULL,
  `different_time_from_creat` float DEFAULT NULL,
  `is_first_setup` int(11) DEFAULT NULL,
  `chronological_number` int(11) NOT NULL,
  PRIMARY KEY (`issue_key`,`chronological_number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `changes_story_points`
--

--
-- Table structure for table `changes_summary`
--

DROP TABLE IF EXISTS `changes_summary_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `changes_summary_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `author` varchar(200) DEFAULT NULL,
  `created` datetime DEFAULT NULL,
  `from_string` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `to_string` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `if_change_first_hour` int(11) DEFAULT NULL,
  `different_time_from_creat` float DEFAULT NULL,
  `is_first_setup` int(11) DEFAULT NULL,
  `is_diff_more_than_ten` int(11) DEFAULT NULL,
  `chronological_number` int(11) NOT NULL,
  # different from from str to to srt (the next after him)
  `ratio_different_char_next` float DEFAULT NULL,
  `ratio_different_word_next` float DEFAULT NULL,
  `num_different_char_minus_next` INT DEFAULT NULL,
  `num_different_char_plus_next` INT DEFAULT NULL,
  `num_different_char_all_next` INT DEFAULT NULL,
  `num_different_word_minus_next` INT DEFAULT NULL,
  `num_different_word_plus_next` INT DEFAULT NULL,
  `num_different_word_all_next` INT DEFAULT NULL,
  # different from from str to current(last) value (the final value)
  `ratio_different_char_last` float DEFAULT NULL,
  `ratio_different_word_last` float DEFAULT NULL,
  `num_different_char_minus_last` INT DEFAULT NULL,
  `num_different_char_plus_last` INT DEFAULT NULL,
  `num_different_char_all_last` INT DEFAULT NULL,
  `num_different_word_minus_last` INT DEFAULT NULL,
  `num_different_word_plus_last` INT DEFAULT NULL,
  `num_different_word_all_last` INT DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`chronological_number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;



DROP TABLE IF EXISTS `changes_criteria_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `changes_criteria_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `author` varchar(200) DEFAULT NULL,
  `created` datetime DEFAULT NULL,
  `from_string` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `to_string` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `if_change_first_hour` int(11) DEFAULT NULL,
  `different_time_from_creat` float DEFAULT NULL,
  `is_first_setup` int(11) DEFAULT NULL,
  `is_diff_more_than_ten` int(11) DEFAULT NULL,
  `chronological_number` int(11) NOT NULL,
  # different from from str to to srt (the next after him)
  `ratio_different_char_next` float DEFAULT NULL,
  `ratio_different_word_next` float DEFAULT NULL,
  `num_different_char_minus_next` INT DEFAULT NULL,
  `num_different_char_plus_next` INT DEFAULT NULL,
  `num_different_char_all_next` INT DEFAULT NULL,
  `num_different_word_minus_next` INT DEFAULT NULL,
  `num_different_word_plus_next` INT DEFAULT NULL,
  `num_different_word_all_next` INT DEFAULT NULL,
  # different from from str to current(last) value (the final value)
  `ratio_different_char_last` float DEFAULT NULL,
  `ratio_different_word_last` float DEFAULT NULL,
  `num_different_char_minus_last` INT DEFAULT NULL,
  `num_different_char_plus_last` INT DEFAULT NULL,
  `num_different_char_all_last` INT DEFAULT NULL,
  `num_different_word_minus_last` INT DEFAULT NULL,
  `num_different_word_plus_last` INT DEFAULT NULL,
  `num_different_word_all_last` INT DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`chronological_number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `changes_summary`
--

--
-- Table structure for table `comments`
--

DROP TABLE IF EXISTS `comments_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `comments_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `author` varchar(200) DEFAULT NULL,
  `id` int(20) NOT NULL,
  `created` datetime DEFAULT NULL,
  `body` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `chronological_number` int(11) DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `comments`
--

--
-- Table structure for table `commits_info`
--

DROP TABLE IF EXISTS `commits_info_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `commits_info_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `author` varchar(200) DEFAULT NULL,
  `insertions` int(11) DEFAULT NULL,
  `code_deletions` int(11) DEFAULT NULL,
  `code_lines` int(11) DEFAULT NULL,
  `files` int(11) DEFAULT NULL,
  `summary` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `message` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `commit` varchar(500) NOT NULL,
  `chronological_number` int(11) DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`commit`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `commits_info`
--

--
-- Table structure for table `components`
--

DROP TABLE IF EXISTS `components_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `components_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `component` varchar(200) NOT NULL,
  `chronological_order` int(11) DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`component`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `components`
--

--
-- Table structure for table `fix_versions`
--

DROP TABLE IF EXISTS `fix_versions_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `fix_versions_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `fix_version` varchar(100) NOT NULL,
  `chronological_number` int(11) DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`fix_version`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `fix_versions`
--

--
-- Table structure for table `issue_links`
--

DROP TABLE IF EXISTS `issue_links_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `issue_links_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `issue_link` varchar(100) NOT NULL,
  `issue_link_name_relation` varchar(200) DEFAULT NULL,
  `chronological_number` int(11) DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`issue_link`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `issue_links`
--


--
-- Table structure for table `labels`
--

DROP TABLE IF EXISTS `labels_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `labels_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `label` varchar(100) NOT NULL,
  `chronological_number` int(11) DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`label`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `labels`
--


--
-- Table structure for table `main_table`
--

DROP TABLE IF EXISTS `main_table_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `main_table_os` (
  `issue_key` char(50) NOT NULL,
  `issue_id` int(20) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `created` datetime NOT NULL,
  `creator` varchar(100) NOT NULL,
  `reporter` varchar(100) NOT NULL,
  `assignee` varchar(200) DEFAULT NULL,
  `date_of_first_response` datetime DEFAULT NULL,
  `epic_link` varchar(100) DEFAULT NULL,
  `issue_type` varchar(45) NOT NULL,
  `last_updated` datetime DEFAULT NULL,
  `priority` varchar(100) DEFAULT NULL,
  `prograss` float DEFAULT NULL,
  `prograss_total` float DEFAULT NULL,
  `resolution` varchar(100) DEFAULT NULL,
  `resolution_date` datetime DEFAULT NULL,
  `status_name` varchar(100) DEFAULT NULL,
  `status_description` varchar(200) DEFAULT NULL,
  `time_estimate` float DEFAULT NULL,
  `time_origion_estimate` float DEFAULT NULL,
  `time_spent` float DEFAULT NULL,
  `attachment` varchar(500) DEFAULT NULL,
  `is_attachment` INT DEFAULT NULL,
  `pull_request_url` varchar(500) DEFAULT NULL,
  `images` varchar(500) DEFAULT NULL,
  `is_images` INT DEFAULT NULL,
  `team` varchar(100) DEFAULT NULL,  
  `story_point` float DEFAULT NULL,
  `summary` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `description` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `acceptance_criteria` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `num_all_changes` int(11) DEFAULT NULL,
  `num_bugs_issue_link` int(11) DEFAULT NULL,
  `num_changes_summary` int(11) DEFAULT NULL,
  `num_changes_description` int(11) DEFAULT NULL,
  `num_changes_acceptance_criteria` int(11) DEFAULT NULL,
  `num_changes_story_point` int(11) DEFAULT NULL,
  `num_comments` int(11) DEFAULT NULL,
  `num_issue_links` int(11) DEFAULT NULL,
  `num_of_commits` int(11) DEFAULT NULL,
  `num_sprints` int(11) DEFAULT NULL,
  `num_sub_tasks` int(11) DEFAULT NULL,
  `num_watchers` int(11) DEFAULT NULL,
  `num_worklog` int(11) DEFAULT NULL,
  `num_versions` int(11) DEFAULT NULL,
  `num_fix_versions` int(11) DEFAULT NULL,
  `num_labels` int(11) DEFAULT NULL,
  `num_components` int(11) DEFAULT NULL,
  `original_summary` mediumtext,
  `num_changes_summary_new` int(11) DEFAULT NULL,
  `original_description` mediumtext,
  `num_changes_description_new` int(11) DEFAULT NULL,
  `original_acceptance_criteria` mediumtext,
  `num_changes_acceptance_criteria_new` int(11) DEFAULT NULL,
  `original_story_points` float DEFAULT NULL,
  `num_changes_story_points_new` int(11) DEFAULT NULL,
  `has_change_summary` int(11) DEFAULT NULL,
  `has_change_description` int(11) DEFAULT NULL,
  `has_change_acceptance_criteria` int(11) DEFAULT NULL,
  `has_change_story_point` int(11) DEFAULT NULL,
  `num_changes_sprint` int(11) DEFAULT NULL,
  `original_summary_description_acceptance` mediumtext,
  `num_changes_summary_description_acceptance` int(11) DEFAULT NULL,
  `has_changes_summary_description_acceptance` int(11) DEFAULT NULL,
  `has_change_summary_description_acceptance_after_sprint` int(11) DEFAULT NULL,
  `has_change_summary_after_sprint` int(11) DEFAULT NULL,
  `has_change_description_after_sprint` int(11) DEFAULT NULL,
  `has_change_acceptance_criteria_after_sprint` int(11) DEFAULT NULL,
  # add feature time_add_to_sprint:
  `time_add_to_sprint` DATETIME DEFAULT NULL,
  # add another features open source:
  `original_summary_sprint` MEDIUMTEXT DEFAULT NULL,
  `num_changes_summary_new_sprint` INT(11) DEFAULT NULL,
  `original_description_sprint` MEDIUMTEXT DEFAULT NULL,
  `num_changes_description_new_sprint` INT(11) DEFAULT NULL,
  `original_acceptance_criteria_sprint` MEDIUMTEXT DEFAULT NULL,
  `num_changes_acceptance_criteria_new_sprint` INT(11) DEFAULT NULL,
  `original_story_points_sprint` FLOAT DEFAULT NULL,
  `num_changes_story_points_new_sprint` INT(11) DEFAULT NULL,
  `has_change_summary_sprint` INT(11) DEFAULT NULL,
  `has_change_description_sprint` INT(11) DEFAULT NULL,
  `has_change_acceptance_criteria_sprint` INT(11) DEFAULT NULL,
  `has_change_story_point_sprint` INT(11) DEFAULT NULL,
  # add the different word count and word ration between before and after sprint
  `different_words_minus_summary` INT(11) DEFAULT NULL,
  `different_words_plus_summary` INT(11) DEFAULT NULL,
  `different_words_minus_description` INT(11) DEFAULT NULL,
  `different_words_plus_description` INT(11) DEFAULT NULL,
  `different_words_minus_acceptance_criteria` INT(11) DEFAULT NULL,
  `different_words_plus_acceptance_criteria` INT(11) DEFAULT NULL,
  `different_words_ratio_all_summary` FLOAT DEFAULT NULL,
  `different_words_ratio_all_description` FLOAT DEFAULT NULL,
  `different_words_ratio_all_acceptance_criteria` FLOAT DEFAULT NULL,
  `different_words_minus_summary_sprint` INT(11) DEFAULT NULL,
  `different_words_plus_summary_sprint` INT(11) DEFAULT NULL,
  `different_words_minus_description_sprint` INT(11) DEFAULT NULL,
  `different_words_plus_description_sprint` INT(11) DEFAULT NULL,
  `different_words_minus_acceptance_criteria_sprint` INT(11) DEFAULT NULL,
  `different_words_plus_acceptance_criteria_sprint` INT(11) DEFAULT NULL,
  `different_words_ratio_all_summary_sprint` FLOAT DEFAULT NULL,
  `different_words_ratio_all_description_sprint` FLOAT DEFAULT NULL,
  `different_words_ratio_all_acceptance_criteria_sprint` FLOAT DEFAULT NULL,
  `num_comments_before_sprint` INT(11) DEFAULT NULL,
  `num_comments_after_sprint` INT(11) DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`issue_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `main_table`
--

--
-- Table structure for table `names_bugs_issue_links`
--

DROP TABLE IF EXISTS `names_bugs_issue_links_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `names_bugs_issue_links_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `bug_issue_link` varchar(100) NOT NULL,
  `chronological_number` int(11) DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`bug_issue_link`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `names_bugs_issue_links`
--


--
-- Table structure for table `sab_task_names`
--

DROP TABLE IF EXISTS `sab_task_names_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `sab_task_names_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `sub_task_name` varchar(200) NOT NULL,
  `chronological_number` int(11) DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`sub_task_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `sab_task_names`
--

--
-- Table structure for table `sprints`
--

DROP TABLE IF EXISTS `sprints_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `sprints_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `sprint_name` varchar(300) NOT NULL,
  `start_date` datetime DEFAULT NULL,
  `end_date` datetime DEFAULT NULL,
  `is_over` int(11) DEFAULT NULL,
  `chronological_number` int(11) DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`sprint_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `sprints`
--

--
-- Table structure for table `versions`
--

DROP TABLE IF EXISTS `versions_os`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `versions_os` (
  `issue_key` char(50) NOT NULL,
  `project_key` varchar(45) NOT NULL,
  `version` varchar(200) NOT NULL,
  `chronological_number` int(11) DEFAULT NULL,
  PRIMARY KEY (`issue_key`,`version`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;



--
-- Dumping data for table `versions`
--