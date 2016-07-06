lazy val root = (project in file(".")).
  settings(
    name := "wlml",
    version := "0.0.1",
    scalaVersion := "2.10.4",
    libraryDependencies  ++= Seq(
      "org.scalanlp" %% "breeze" % "0.12"
    ),
    resolvers ++= Seq(
      "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
    )
  )
